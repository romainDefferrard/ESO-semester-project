from utils.flight_data import FlightData
from utils.raster_loader import RasterLoader
from utils.footprint_generator import Footprint
from utils.patch_generator import PatchGenerator
from utils.las_extractor import LasExtractor
from utils.gui import GUIMainWindow

from PyQt6.QtWidgets import QApplication
import sys
import os
import time
from multiprocessing import Manager, Pool
import multiprocessing as mp
import logging
import yaml
from rasterio.plot import show
import argparse

import concurrent.futures

# Configuration
parser = argparse.ArgumentParser()
parser.add_argument("--yml", "-y", required=True, help="Path to the configuration file")
args = parser.parse_args()

# Config paths
config = yaml.safe_load(open(args.yml, "r"))
OUTPUT_DIR = config["OUTPUT_DIR"]
LAS_DIR = config["LAS_DIR"]


# Logger setup
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def load_data():
    # load flights, raster and compute footprints
    time0 = time.time()
    logging.info("Loading flight and raster data...")

    # Load flight data
    flight_manager = FlightData(config)
    flight_bounds = flight_manager.bounds
    flights = flight_manager.flights

    # Load raster data
    raster_loader = RasterLoader(config, flight_bounds=flight_bounds)
    raster = raster_loader.raster
    raster_mesh = (raster_loader.x_mesh, raster_loader.y_mesh)
    logging.info(f"Flights and raster loaded in {time.time() - time0:.2f}s.")

    time1 = time.time()
    # Compute footprint
    footprint = Footprint(raster=raster, raster_mesh=raster_mesh, flights=flights, config=config)
    logging.info(f"Footprint generation in {time.time() - time1:.2f}s. ({len(flights)} flights)")
    return raster_loader, footprint


def run_gui(footprint, patches_all, centerlines_all, contours_all, raster, raster_mesh):
    """Launch the GUI and return the updated state."""
    app = QApplication(sys.argv)
    window = GUIMainWindow(
        superpositions=footprint.superpos_masks,
        patches=patches_all,
        centerlines=centerlines_all,
        patch_params=config["PATCH_DIMS"],
        raster_mesh=raster_mesh,
        raster=raster,
        contours=contours_all,
        extraction_state=False,
        flight_pairs=footprint.superpos_flight_pairs,
        output_dir=OUTPUT_DIR,
    )
    window.show()
    app.exec()

    return window.control_panel.extraction_state, window.control_panel.new_patches_poly


def tasks_extraction(footprint, patches_poly, LAS_DIR, OUTPUT_DIR):
    tasks = [(flight_i, flight_j, footprint, patches_poly, LAS_DIR, OUTPUT_DIR) for flight_i, flight_j in footprint.superpos_flight_pairs]
    return tasks


def process_flight(flight_id, flight_patch, LAS_DIR, OUTPUT_DIR, pair_dir):
    """Process a single flight."""
    # logging.info(f"Processing Flight {flight_id}...")
    input_file = LAS_DIR.format(flight_id=flight_id)
    extractor = LasExtractor(input_file, flight_patch)

    if extractor.read_point_cloud():
        extractor.process_all_patches(flight_patch, OUTPUT_DIR, flight_id, pair_dir)

    # logging.info(f"Flight {flight_id} extraction completed.")


def extract_flight_pair(flight_i, flight_j, footprint, patches_poly, LAS_DIR, OUTPUT_DIR):
    """Extract patches for a given flight pair using thread-based parallel processing."""
    flight_patch = patches_poly[footprint.superpos_flight_pairs.index((flight_i, flight_j))]

    # Create the directory for this flight pair
    pair_dir = f"{OUTPUT_DIR}/Flights_{flight_i}_{flight_j}"
    os.makedirs(pair_dir, exist_ok=True)

    # Pool ne peut pas avoir d'enfant donc -> ThreadPoolExecutor on gagne pas énorme ~10s
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(process_flight, flight_i, flight_patch, LAS_DIR, OUTPUT_DIR, pair_dir)
        executor.submit(process_flight, flight_j, flight_patch, LAS_DIR, OUTPUT_DIR, pair_dir)

    logging.info(f"Extraction completed for flights {flight_i} and {flight_j}.")


def run_extraction(footprint, patches_poly, LAS_DIR, OUTPUT_DIR):
    """Exécute l'extraction des patches pour chaque paire de vols et sauvegarde les résultats dans des dossiers distincts."""
    logging.info(f"Extracting patches for all flight pairs...{footprint.superpos_flight_pairs}")
    time0 = time.time()

    # Generate extraction tasks
    tasks = tasks_extraction(footprint, patches_poly, LAS_DIR, OUTPUT_DIR)

    # Use multiprocessing to speed up extraction
    num_outer_processes = max(1, mp.cpu_count() // 2)  # Use half of the available CPUs du to embedded parallelism of operations
    with Pool(processes=num_outer_processes) as pool:
        pool.starmap(extract_flight_pair, tasks)

    logging.info(f"Extraction in {time.time() - time0:.2f}s.")


def main():
    # Load data
    raster_loader, footprint = load_data()
    raster, raster_mesh = raster_loader.raster, (raster_loader.x_mesh, raster_loader.y_mesh)

    # Create PatchGenerator instance with all superposition zones
    patch_gen = PatchGenerator(superpos_zones=footprint.superpos_masks, 
                               raster_mesh=raster_mesh, 
                               raster=raster, 
                               patch_params=config["PATCH_DIMS"])

    # Now, patch_gen contains all the patches and centerlines
    patches_all = patch_gen.patches_list
    centerlines_all = patch_gen.centerlines_list  # All centerlines
    contours_all = patch_gen.contours_list  # All contours

    # Run GUI with all patches, centerlines, and contours
    extraction_state, new_patches_poly = run_gui(footprint, patches_all, centerlines_all, contours_all, raster, raster_mesh)
    if extraction_state:
        run_extraction(footprint, new_patches_poly, LAS_DIR, OUTPUT_DIR)
    else:
        logging.info("Window closed without extraction.")


if __name__ == "__main__":
    main()

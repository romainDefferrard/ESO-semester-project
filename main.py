import argparse
import concurrent.futures
import logging
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from typing import List, Tuple
import numpy as np
import yaml
from PyQt6.QtWidgets import QApplication

from utils.flight_data import FlightData
from utils.footprint_generator import Footprint
from utils.gui import GUIMainWindow
from utils.las_extractor import LasExtractor
from utils.patch_generator import PatchGenerator
from utils.patch_model import Patch
from utils.raster_loader import RasterLoader

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

def load_data() -> Tuple[RasterLoader, Footprint]:
    # load flights, raster and compute footprints
    time0 = time.time()
    logging.info("Loading flight and raster data...")

    # Load flight data
    fd = FlightData(config)

    # Load raster data
    rl = RasterLoader(config, flight_bounds=fd.bounds)
    raster = rl.raster
    raster_mesh = (rl.x_mesh, rl.y_mesh)
    
    logging.info(f"Flights and raster loaded in {time.time() - time0:.2f}s.")
    time1 = time.time()
    
    # Compute footprint
    footprint = Footprint(raster=raster, raster_mesh=raster_mesh, flights=fd.flights, config=config)
    logging.info(f"Footprint generation in {time.time() - time1:.2f}s. ({len(fd.flights)} flights)")
    
    return raster, raster_mesh, footprint


def run_gui(footprint: Footprint, patches_all: List, centerlines_all: List, contours_all: List, 
            raster: np.ndarray, raster_mesh: Tuple[np.ndarray, np.ndarray]) -> Tuple[bool, List]:

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

    return window.control_panel.extraction_state, window.control_panel.new_patches_instance


def tasks_extraction(footprint: Footprint, patches: List[List[Patch]], LAS_DIR: str, OUTPUT_DIR: str) -> List[Tuple[str, str, Footprint, List, str, str]]:
    return [(flight_i, flight_j, footprint, patches, LAS_DIR, OUTPUT_DIR) for flight_i, flight_j in footprint.superpos_flight_pairs]


def process_flight(flight_id: str, flight_patch: List, LAS_DIR: str, OUTPUT_DIR: str, pair_dir: str) -> None:
    """Process a single flight."""

    logging.info(f"Processing Flight {flight_id}...")
    input_file = LAS_DIR.format(flight_id=flight_id)
    extractor = LasExtractor(config, input_file, flight_patch)

    if extractor.read_point_cloud():
        extractor.process_all_patches(flight_patch, OUTPUT_DIR, flight_id, pair_dir)

    # logging.info(f"Flight {flight_id} extraction completed.")


def extract_flight_pair(flight_i: str, flight_j: str, footprint: Footprint, patches: List[List[Patch]], LAS_DIR: str, OUTPUT_DIR: str) -> None:
    """Extract patches for a given flight pair using thread-based parallel processing."""
    pair_index = footprint.superpos_flight_pairs.index((flight_i, flight_j))
    patch_group = patches[pair_index]

    pair_dir = f"{OUTPUT_DIR}/Flights_{flight_i}_{flight_j}"
    os.makedirs(pair_dir, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(process_flight, flight_i, patch_group, LAS_DIR, OUTPUT_DIR, pair_dir)
        executor.submit(process_flight, flight_j, patch_group, LAS_DIR, OUTPUT_DIR, pair_dir)

    logging.info(f"Extraction completed for flights {flight_i} and {flight_j}.")



def run_extraction(footprint: Footprint, patches: List[List[Patch]], LAS_DIR: str, OUTPUT_DIR: str) -> None:
    """Exécute l'extraction des patches pour chaque paire de vols et sauvegarde les résultats dans des dossiers distincts."""
    logging.info(f"Extracting patches for all flight pairs...{footprint.superpos_flight_pairs}")
    time0 = time.time()
    if config["EXTRACTION_MODE"] == "encoded":
        # --- ENCODED: one file per flight, check all patches per flight ---
        all_patches_flat = [patch for group in patches for patch in group]

        flight_ids = sorted(set(f for pair in footprint.superpos_flight_pairs for f in pair))

        tasks = []
        for flight_id in flight_ids:
            pair_dir = f"{OUTPUT_DIR}/Flight_{flight_id}"
            os.makedirs(pair_dir, exist_ok=True)
            tasks.append((flight_id, all_patches_flat, LAS_DIR, OUTPUT_DIR, pair_dir))

        # Parallel processing per flight
        with Pool(processes=mp.cpu_count()) as pool:
            pool.starmap(process_flight, tasks)

    else:
        # --- INDEPENDENT: two file per patches, one for each flight concerned ---
        tasks = tasks_extraction(footprint, patches, LAS_DIR, OUTPUT_DIR)
        num_outer_processes = max(1, mp.cpu_count() // 2) # limit CPUs, embedded parallelism inside 

        with Pool(processes=num_outer_processes) as pool:
            pool.starmap(extract_flight_pair, tasks)

    logging.info(f"Extraction completed in {time.time() - time0:.2f}s.")


def main() -> None:
    # Load data
    raster, raster_mesh, footprint = load_data()
    # Create PatchGenerator instance with all superposition zones
    pg = PatchGenerator(superpos_zones=footprint.superpos_masks, raster_mesh=raster_mesh, raster=raster, patch_params=config["PATCH_DIMS"])

    # Now, pg contains all the patches instances, centerlines and contours 
    patches = pg.patches_list
    centerlines = pg.centerlines_list  # All centerlines
    contours = pg.contours_list  # All contours

    # Run GUI with all patches, centerlines, and contours
    extraction_state, new_patches_instance = run_gui(footprint, patches, centerlines, contours, raster, raster_mesh)
    if extraction_state:
        run_extraction(footprint, new_patches_instance, LAS_DIR, OUTPUT_DIR)
    else:
        logging.info("Window closed without extraction.")


if __name__ == "__main__":
    main()

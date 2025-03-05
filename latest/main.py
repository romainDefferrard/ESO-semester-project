"""Notes d'utilisation: 
- noms des .las de: nommer de 1 à XX (éviter 01, 02, ...)
"""

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
import multiprocessing
import logging
import yaml
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show


# Configuration 
DATASET = "Config/Arpette_config.yaml"

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
# Load the dataset configuration
config = load_config(DATASET)

# Config paths 
#GPSTIME_PATH = config["GPSTIME_PATH"]
TRAJECTORY_PATH = config["TRAJECTORY_PATH"]
MNT_PATH = config["MNT_PATH"]
OUTPUT_DIR = config["OUTPUT_DIR"]
#LAS_NAME = config["LAS_PATTERN"]

LAS_DIR = config["LAS_DIR"]
LOG_DIR = config["LOG_DIR"]

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

def load_data():
    # load flights, raster and compute footprints
    logging.info("Loading flight and raster data...")

    # Load flight data
    flight_manager = FlightData(LAS_DIR, LOG_DIR, TRAJECTORY_PATH)
    flight_bounds = flight_manager.bounds
    flights = flight_manager.flights
    
    # Load raster data
    raster_loader = RasterLoader(MNT_PATH, epsg=2056, flight_bounds=flight_bounds)
    raster = raster_loader.raster
    raster_mesh = (raster_loader.x_mesh, raster_loader.y_mesh)

    # Compute footprint
    footprint = Footprint(raster=raster, raster_mesh=raster_mesh, flights=flights)

    return raster_loader, footprint


def create_tasks(footprint, patches_poly):
    """Generate tasks for multiprocessing."""
    tasks = []
    for idx, (flight_i, flight_j) in enumerate(footprint.superpos_flight_pairs):
        flight_patch = patches_poly[idx]
        pair_dir = f"{OUTPUT_DIR}/Flights_{flight_i}_{flight_j}"
        os.makedirs(pair_dir, exist_ok=True)

        for patch_idx, patch in enumerate(flight_patch):  
            tasks.append((flight_i, patch_idx, patch, pair_dir, LAS_DIR))
            tasks.append((flight_j, patch_idx, patch, pair_dir, LAS_DIR))

    return tasks


def run_gui(footprint, patches_all, centerlines_all, contours_all, raster, raster_mesh):
    """Launch the GUI and return the updated state."""
    app = QApplication(sys.argv)
    window = GUIMainWindow(
        superpositions=footprint.superpos_masks,
        patches=patches_all,
        centerlines=centerlines_all,
        patch_params=(50, 100, 400),
        raster_mesh=raster_mesh,
        raster=raster,
        contours=contours_all,
        extraction_state=False,
        flight_pairs=footprint.superpos_flight_pairs
    )
    window.show()
    app.exec()

    return window.control_panel.extraction_state, window.control_panel.new_patches_poly

def extract_patch(flight_id, patch_idx, patch, pair_dir, las_pattern):
    """Extract LAS data for a given flight and patch."""
    input_file = las_pattern.format(flight_id=flight_id)  # Use the pattern from the config
    output_file = f"{pair_dir}/patch_{patch_idx}_flight_{flight_id}.laz"
    flight_pair = os.path.basename(pair_dir)

    logging.info(f"Extracting patch {patch_idx} for flight {flight_id} ({flight_pair})")
    LasExtractor(input=input_file, output=output_file, patch=patch)
    
def run_extraction(tasks):
    """Extraction of patches in corresponding flights .las"""
    if not tasks:
        logging.info("No tasks to execute.")
        return

    logging.info(f"Starting LAS extraction for {len(tasks)} patches...")
    start_time = time.time()

    with multiprocessing.Pool() as pool:
        pool.starmap(extract_patch, tasks)

    logging.info(f"Extraction completed in {time.time() - start_time:.2f} seconds.")



def main():
    raster_loader, footprint = load_data()
    raster, raster_mesh = raster_loader.raster, (raster_loader.x_mesh, raster_loader.y_mesh)

    # Create PatchGenerator instance with all superposition zones
    patch_gen = PatchGenerator(superpos_zones=footprint.superpos_masks, 
                                raster_mesh=raster_mesh, 
                                raster=raster, 
                                patch_params=(50, 100, 400))
    
    # Now, patch_gen contains all the patches and centerlines
    patches_all = patch_gen.patches_list
    centerlines_all = patch_gen.centerlines_list  # All centerlines
    contours_all = patch_gen.contours_list  # All contours

    # Run GUI with all patches, centerlines, and contours
    extraction_state, new_patches_poly = run_gui(footprint, patches_all, centerlines_all, contours_all, raster, raster_mesh)

    if extraction_state:
        tasks = create_tasks(footprint, new_patches_poly)
        run_extraction(tasks)
    else:
        logging.info("Window closed without extraction.")



if __name__ == "__main__":
    main()

"""Notes de modif à faire: 
- modifier mode de centerline parmis une liste (voir résultats du val d'Arpette)
- enlever le patchgenerator du main et le garder juste de l'UI

QUESTION:
- est-ce que ça te va si t'as une bande mais qu'on garde le KNN comme ça c'est pas 
trop long à compute ou alors tu veux un truc vraiment carré
    - pas de perte dans la dimension temporelle (along axis)

"""
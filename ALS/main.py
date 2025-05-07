
import argparse
import concurrent.futures
import logging
import multiprocessing as mp
import os
import sys
import time
from multiprocessing import Pool
from typing import List, Tuple
import numpy as np
import yaml
from shapely.geometry import Polygon
from shapely.ops import unary_union
import cv2
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
MULTIPROCESSING = config["MULTIPROCESSING"]

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

def load_data() -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], Footprint]:
    time0 = time.time()
    logging.info("Loading flight and raster data...")

    fd = FlightData(config)
    rl = RasterLoader(config, flight_bounds=fd.bounds)
    raster = rl.raster
    raster_mesh = (rl.x_mesh, rl.y_mesh)

    logging.info(f"Flights and raster loaded in {time.time() - time0:.2f}s.")
    time1 = time.time()

    footprint = Footprint(raster=raster, raster_mesh=raster_mesh, flights=fd.flights, config=config)
    logging.info(f"Footprint generation in {time.time() - time1:.2f}s. ({len(fd.flights)} flights)")

    return raster, raster_mesh, footprint

def run_gui(footprint: Footprint, patches_all: List, centerlines_all: List, contours_all: List,
            raster: np.ndarray, raster_mesh: Tuple[np.ndarray, np.ndarray]) -> Tuple[bool, List, str]:

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

    return window.control_panel.extraction_state, window.control_panel.new_patches_instance, window.control_panel.output_dir

def get_flight_union_contour(flight_id: str, superpos_pairs: List[Tuple[str, str]], contours_list: List[np.ndarray]) -> Polygon:
    """ vu que les patches sont uniquement généré dans les zones d'overlap, on check ces zones et on comparera avec les patch pour filter 
    uniquement ceux qui sont utiles pour nous. """
    polygons = []
    for i, (f1, f2) in enumerate(superpos_pairs):
        if flight_id in (f1, f2):
            contour = contours_list[i]
            if isinstance(contour, np.ndarray):
                coords = contour.reshape(-1, 2)
                if len(coords) >= 3:
                    polygons.append(Polygon(coords))
            elif isinstance(contour, Polygon):
                polygons.append(contour)
    return unary_union(polygons) if polygons else Polygon()

def tasks_extraction(footprint: Footprint, patches: List[List[Patch]], LAS_DIR: str, OUTPUT_DIR: str) -> List[Tuple[str, str, Footprint, List, str, str]]:
    return [(flight_i, flight_j, footprint, patches, LAS_DIR, OUTPUT_DIR) for flight_i, flight_j in footprint.superpos_flight_pairs]

def process_flight(flight_id: str, flight_patch: List[Patch], LAS_DIR: str, OUTPUT_DIR: str, pair_dir: str,
                   superpos_pairs: List[Tuple[str, str]], contours_all: List[np.ndarray]) -> None:
    time0 = time.time()
    #logging.info(f"Processing Flight {flight_id}...")
    input_file = LAS_DIR.format(flight_id=flight_id)

    flight_polygon = get_flight_union_contour(flight_id, superpos_pairs, contours_all)

    relevant_patches = []
    for patch in flight_patch:
        patch_poly = patch.shapely_polygon
        if patch_poly.intersects(flight_polygon):
            relevant_patches.append(patch)

    extractor = LasExtractor(config, input_file, relevant_patches)
    if extractor.read_point_cloud():
        extractor.process_all_patches(relevant_patches, OUTPUT_DIR, flight_id, pair_dir)
    logging.info(f"Flight {flight_id} done: {time.time()-time0:.2f}s (Reading+Encoding+Writing)")
    
    
def extract_flight_pair(flight_i: str, flight_j: str, footprint: Footprint, patches: List[List[Patch]], LAS_DIR: str, OUTPUT_DIR: str) -> None:
    pair_index = footprint.superpos_flight_pairs.index((flight_i, flight_j))
    patch_group = patches[pair_index]
    pair_dir = f"{OUTPUT_DIR}/Flights_{flight_i}_{flight_j}"
    os.makedirs(pair_dir, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(process_flight, flight_i, patch_group, LAS_DIR, OUTPUT_DIR, pair_dir, footprint.superpos_flight_pairs, [])
        executor.submit(process_flight, flight_j, patch_group, LAS_DIR, OUTPUT_DIR, pair_dir, footprint.superpos_flight_pairs, [])

    logging.info(f"Extraction completed for flights {flight_i} and {flight_j}.")

def run_extraction(footprint: Footprint, patches: List[List[Patch]], LAS_DIR: str, OUTPUT_DIR: str,
                   contours_all: List[np.ndarray]) -> None:
    logging.info("Extracting patches...")
    time0 = time.time()

    if config["EXTRACTION_MODE"] == "Extra_Bytes":
        all_patches_flat = [patch for group in patches for patch in group]
        flight_ids = sorted(set(f for pair in footprint.superpos_flight_pairs for f in pair))

        tasks = []
        for flight_id in flight_ids:
            pair_dir = f"{OUTPUT_DIR}/Flight_{flight_id}"
            os.makedirs(pair_dir, exist_ok=True)
            tasks.append((flight_id, all_patches_flat, LAS_DIR, OUTPUT_DIR, pair_dir, footprint.superpos_flight_pairs, contours_all))

        if MULTIPROCESSING:
            with Pool(processes=2) as pool: # mp.cpu_count()
                pool.starmap(process_flight, tasks)
        else:
            for args in tasks:
                process_flight(*args)
    else:
        tasks = tasks_extraction(footprint, patches, LAS_DIR, OUTPUT_DIR)
        num_outer_processes = max(1, mp.cpu_count() // 2)
        with Pool(processes=num_outer_processes) as pool:
            pool.starmap(extract_flight_pair, tasks)

    logging.info(f"Extraction completed in {time.time() - time0:.2f}s.")

def main() -> None:
    raster, raster_mesh, footprint = load_data()
    pg = PatchGenerator(superpos_zones=footprint.superpos_masks, raster_mesh=raster_mesh, raster=raster, patch_params=config["PATCH_DIMS"])
    extraction_state, new_patches_instance, updated_output_dir = run_gui(footprint, pg.patches_list, pg.centerlines_list, pg.contours_list, raster, raster_mesh)

    if extraction_state:
        run_extraction(footprint, new_patches_instance, LAS_DIR, updated_output_dir, pg.contours_list)
    else:
        logging.info("Window closed without extraction.")

if __name__ == "__main__":
    main()

import argparse
import multiprocessing as mp
import os
import sys
from typing import List, Tuple
import numpy as np
import yaml
from shapely.geometry import Polygon
from shapely.ops import unary_union
from PyQt6.QtWidgets import QApplication
from tqdm import tqdm
import logging

from utils.flight_data import FlightData
from utils.footprint_generator import Footprint
from utils.gui import GUIMainWindow
from utils.las_extractor import LasExtractor
from utils.patch_generator import PatchGenerator
from utils.patch_model import Patch
from utils.raster_loader import RasterLoader
from utils.timer_logger import TimerLogger




class ALSPipeline():
    def __init__(self, config_path: str):
        self.config = yaml.safe_load(open(config_path, "r"))
        self.las_dir = self.config["LAS_DIR"]
        self.timer = TimerLogger()
        
        
    def load_data(self) -> None:
        self.timer.start("Flight & raster loading")
        fd = FlightData(self.config)
        rl = RasterLoader(self.config, flight_bounds=fd.bounds)
        self.raster = rl.raster
        self.raster_mesh = (rl.x_mesh, rl.y_mesh)
        self.timer.stop("Flight & raster loading")

        self.timer.start("Footprint generation")
        self.footprint = Footprint(raster=self.raster, raster_mesh=self.raster_mesh, flights=fd.flights, config=self.config)
        self.timer.stop("Footprint generation")


    def generate_patches(self):
        self.timer.start("Patch generation")
        self.pg = PatchGenerator(superpos_zones=self.footprint.superpos_masks, raster_mesh=self.raster_mesh, raster=self.raster, patch_params=self.config["PATCH_DIMS"])
        self.timer.stop("Patch generation")

    def launch_gui(self):
        app = QApplication(sys.argv)
        window = GUIMainWindow(
            superpositions=  self.footprint.superpos_masks,
            patches=         self.pg.patches_list,
            centerlines=     self.pg.centerlines_list,
            patch_params=    self.config["PATCH_DIMS"],
            raster_mesh=     self.raster_mesh,
            raster=          self.raster,
            contours=        self.pg.contours_list,
            extraction_state=False,
            flight_pairs=    self.footprint.superpos_flight_pairs,
            output_dir=      self.config["OUTPUT_DIR"],
        )
        window.show()
        app.exec()
        
        self.extraction_state = window.control_panel.extraction_state
        self.patch_list = window.control_panel.new_patches_instance
        self.output_dir = window.control_panel.output_dir 
        
    def extract(self):
        self.timer.start("Patch extraction (all flights)")
        
        if self.config["EXTRACTION_MODE"] in ["Extra_Bytes", "binary"]:
            all_patches_flat = [patch for group in self.patch_list for patch in group]
            flight_ids = sorted(set(f for pair in self.footprint.superpos_flight_pairs for f in pair))

            tasks = []
            for flight_id in flight_ids:
                pair_dir = f"{self.output_dir}/Flight_{flight_id}"
                tasks.append((flight_id, all_patches_flat, self.config["LAS_DIR"], self.output_dir, pair_dir, self.footprint.superpos_flight_pairs, self.pg.contours_list))

            for args in tqdm(tasks, desc="Extracting patches per flight", unit="flight"):
                self.process_flight(*args)
            
        elif self.config["EXTRACTION_MODE"] == "independent":
            for (flight_i, flight_j), patch_group in zip(self.footprint.superpos_flight_pairs, self.patch_list):
                for flight_id in [flight_i, flight_j]:
                    pair_dir = f"{self.output_dir}/Flights_{flight_i}_{flight_j}"
                    os.makedirs(pair_dir, exist_ok=True)
                    self.process_flight(flight_id, patch_group, self.config["LAS_DIR"], self.output_dir, pair_dir, self.footprint.superpos_flight_pairs, self.pg.contours_list)

        self.timer.start("Patch extraction (all flights)")

    def process_flight(self, flight_id: str, flight_patch: List[Patch], LAS_DIR: str, OUTPUT_DIR: str,
                        pair_dir: str, superpos_pairs: List[Tuple[str, str]], contours_all: List[np.ndarray]) -> None:
        input_file = LAS_DIR.format(flight_id=flight_id)
        flight_polygon = self.get_flight_union_contour(flight_id, superpos_pairs, contours_all)

        relevant_patches = [patch for patch in flight_patch if patch.shapely_polygon.intersects(flight_polygon)]

        extractor = LasExtractor(self.config, input_file, relevant_patches)
        if extractor.read_point_cloud():
            extractor.process_all_patches(relevant_patches, OUTPUT_DIR, flight_id, pair_dir)

    def get_flight_union_contour(self, flight_id: str, superpos_pairs: List[Tuple[str, str]], contours_list: List[np.ndarray]) -> Polygon:
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

    def run(self):
        self.timer.start("ALS total time")
        self.load_data()
        self.generate_patches()
        self.launch_gui()
        
        if self.extraction_state:
            self.extract()
        
        self.timer.stop("ALS total time")
        self.timer.summary()
        
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--yml", "-y", required=True, help="Path to the configuration file")
    args = parser.parse_args()

    pipeline = ALSPipeline(config_path=args.yml)
    pipeline.run()
        

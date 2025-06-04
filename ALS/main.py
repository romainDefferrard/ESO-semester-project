"""
Filename: main.py
Author: Romain Defferrard
Date: 04-06-2025

Description:
    Main pipeline for Airborne Laser Scanning (ALS) data processing.
    Handles data loading, footprint generation, patch creation, GUI interaction,
    and LAS file extraction based on patch overlap with estimated footprints.
    
    This pipeline uses a TimerLogger utility to benchmark steps in the pipeline.

"""

import argparse
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
        """
        Initialize the ALS pipeline using the provided YAML configuration.

        Input:
            config_path (str): Path to the YAML configuration file

        Output:
            None
        """
        self.config = yaml.safe_load(open(config_path, "r"))
        self.las_dir = self.config["LAS_DIR"]
        self.timer = TimerLogger()
    
    def load_data(self) -> None:
        """
        Load flight metadata and raster elevation tiles using the config file.

        Output:
            None
        """
        self.timer.start("Flight & raster loading")
        fd = FlightData(self.config)
        rl = RasterLoader(self.config, flight_bounds=fd.bounds)
        self.raster = rl.raster
        self.raster_mesh = (rl.x_mesh, rl.y_mesh)
        self.flight_data = fd  
        self.timer.stop("Flight & raster loading")


    def generate_footprint(self) -> None:
        """
        Generate the estimated LiDAR footprints for each flight.

        Output:
            None
        """
        self.timer.start("Footprint generation")
        self.footprint = Footprint(
            raster=self.raster,
            raster_mesh=self.raster_mesh,
            flights=self.flight_data.flights,
            config=self.config
        )
        self.timer.stop("Footprint generation")


    def generate_patches(self):
        """
        Generate rectangular patches from overlapping zones in LiDAR footprints.

        Output:
            None
        """
        self.timer.start("Patch generation")
        self.pg = PatchGenerator(superpos_zones=self.footprint.superpos_masks, raster_mesh=self.raster_mesh, patch_params=self.config["PATCH_DIMS"])
        self.timer.stop("Patch generation")

    def launch_gui(self):
        """
        Launch the PyQt6 GUI for patch visualization and selection.

        Output:
            None
        """
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
        """
        Extract points for each relevant patch across all flights based on the extraction mode.

        Output:
            None
        """
        self.timer.start("Patch extraction (all flights)")

        if self.config["EXTRACTION_MODE"] in ["Extra_Bytes", "binary"]:
            all_patches_flat = [patch for group in self.patch_list for patch in group]
            flight_ids = sorted(set(f for pair in self.footprint.superpos_flight_pairs for f in pair))

            tasks = []
            for flight_id in flight_ids:
                pair_dir = f"{self.output_dir}/Flight_{flight_id}"
                tasks.append((
                    flight_id, all_patches_flat, self.config["LAS_DIR"], self.output_dir,
                    pair_dir, self.footprint.superpos_flight_pairs, self.pg.contours_list
                ))

            for args in tqdm(tasks, desc="Extracting patches per flight", unit="flight"):
                self.process_flight(*args)

        elif self.config["EXTRACTION_MODE"] == "independent":
            for (flight_i, flight_j), patch_group in zip(self.footprint.superpos_flight_pairs, self.patch_list):
                for flight_id in [flight_i, flight_j]:
                    pair_dir = f"{self.output_dir}/Flights_{flight_i}_{flight_j}"
                    os.makedirs(pair_dir, exist_ok=True)
                    self.process_flight(
                        flight_id, patch_group, self.config["LAS_DIR"], self.output_dir,
                        pair_dir, self.footprint.superpos_flight_pairs, self.pg.contours_list
                    )

        self.timer.stop("Patch extraction (all flights)")


        

    def process_flight(self, flight_id: str, flight_patch: List[Patch], LAS_DIR: str, OUTPUT_DIR: str,
                        pair_dir: str, superpos_pairs: List[Tuple[str, str]], contours_all: List[np.ndarray]) -> None:
        """
        Processes a single flight by filtering relevant patches, loading the point cloud,
        and applying the configured extraction method. Relevant patches are defined as those whose polygon intersects
        the union polygon of the flight's observed footprint (i.e., merged contours from overlapping flight pairs 
        involving this flight ID).

        Inputs:
            flight_id (str): Flight identifier.
            flight_patch (List[Patch]): Patches associated with the current flight.
            LAS_DIR (str): Format string to locate LAS file from flight ID.
            OUTPUT_DIR (str): Base output directory.
            pair_dir (str): Output directory for this specific flight pair.
            superpos_pairs (List[Tuple[str, str]]): All overlapping flight pairs.
            contours_all (List[np.ndarray]): Corresponding list of contours.

        Output:
            None
        """
        
        input_file = LAS_DIR.format(flight_id=flight_id)
        flight_polygon = self.get_flight_union_contour(flight_id, superpos_pairs, contours_all)

        relevant_patches = [patch for patch in flight_patch if patch.shapely_polygon.intersects(flight_polygon)]

        extractor = LasExtractor(self.config, input_file, relevant_patches)
        if extractor.read_point_cloud():
            extractor.process_all_patches(relevant_patches, OUTPUT_DIR, flight_id, pair_dir)
            

    def get_flight_union_contour(self, flight_id: str, superpos_pairs: List[Tuple[str, str]], contours_list: List[np.ndarray]) -> Polygon:
        """
        Return the union of all contours (polygons) involving the given flight ID.

        Input:
            flight_id (str): Flight identifier.
            superpos_pairs (List[Tuple[str, str]]): Pairs of overlapping flights.
            contours_list (List[np.ndarray]): Contours corresponding to each pair.

        Output:
            Polygon: Merged polygon of all overlapping contours for the flight.
        """
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
        """
        Execute the full ALS pipeline from loading data to extraction.

        Output:
            None
        """
        self.timer.start("ALS total time")
        self.load_data()
        self.generate_footprint()
        self.generate_patches()
        self.launch_gui()
        
        if self.extraction_state: # Unlocked in GUI
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

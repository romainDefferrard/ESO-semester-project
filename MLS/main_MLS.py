"""
Filename: main_MLS.py
Author: Romain Defferrard
Date: 04-06-2025

Description:
    Pipeline for Mobile Laser Scanning (MLS) segment intersection analysis.
    Loads shapefile data, adds buffer zones to each segment, detects and filter overlapping
    segments based on shared geometry, and visualizes results via a GUI interface.
    This pipeline uses a TimerLogger utility to benchmark steps in the pipeline.
"""
import geopandas as gpd
import warnings
import argparse
import yaml
import sys
import logging


from PyQt6.QtWidgets import QApplication
from utils.GUI_MLS import GUI_MLS
from utils.timer_logger import TimerLogger

class MLSPipeline:
    def __init__(self, config_path):
        """
        Initialize the MLS pipeline from a YAML config file.

        Input:
            config_path (str): Path to the YAML configuration file.

        Output:
            None
        """
        self.config = yaml.safe_load(open(config_path, "r"))
        self.threshold = 2.5 * self.config["BUFFER"] # Empirically found
        self.timer = TimerLogger()
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    def pipeline(self):
        """
        Run the full MLS pipeline: 
            - load shapefile, 
            - buffer geometries
            - compute intersections
            - and launch GUI.

        Output:
            None
        """
        self.timer.start("Total pipeline time")
        self.timer.start("File loading")
        self.open_file()
        self.timer.stop("File loading")

        self.timer.start("Buffering")
        self.add_buffer()
        self.timer.stop("Buffering")

        self.timer.start("Intersection detection")
        self.intersections = self.get_intersections()
        self.timer.stop("Intersection detection")

        self.run_gui()
        self.timer.stop("Total pipeline time")

        self.timer.summary()

    def run_gui(self):
        """
        Launch the PyQt6 GUI with the segments and intersections.

        Output:
            None
        """
        app = QApplication(sys.argv)
        window = GUI_MLS(
            gdf=self.gdf,
            intersections=self.intersections,
            output=self.config["OUTPUT_PATH"],
        )
        window.show()
        app.exec()

    def open_file(self):
        """
        Load the input shapefile and convert it to the target CRS.

        Output:
            None
        """
        self.gdf = gpd.read_file(self.config["SHP_PATH"])
        self.gdf = self.gdf.to_crs(epsg=self.config["EPSG"])

    def add_buffer(self):
        """
        Add a buffer polygon around each geometry in the shapefile.

        Output:
            None
        """
        self.gdf["buffer"] = self.gdf.geometry.buffer(self.config["BUFFER"])

    def get_intersections(self):
        """
        Compute all intersections between buffered segments.

        Filters out very short overlaps and end-to-end connections.
        
        Output:
            GeoDataFrame: Contains valid segment intersections.
        """
        records = []
        for i in range(len(self.gdf)):
            for j in range(i + 1, len(self.gdf)):
                buf1 = self.gdf.loc[i, "buffer"]
                buf2 = self.gdf.loc[j, "buffer"]

                if buf1.intersects(buf2):
                    overlap_poly = buf1.intersection(buf2)

                    if not overlap_poly.is_empty:
                        line1 = self.gdf.loc[i, "geometry"].intersection(overlap_poly)
                        line2 = self.gdf.loc[j, "geometry"].intersection(overlap_poly)
                        shared_line = line1.union(line2)
                        if shared_line.is_empty:
                            continue
                        shared_length = shared_line.length
                        if shared_length < self.threshold:
                            continue

                        records.append({
                            "id_1": self.gdf.loc[i, "id"],
                            "id_2": self.gdf.loc[j, "id"],
                            "overlap_geom": overlap_poly,
                            "shared_length_m": shared_length,
                            "overlap_area_m2": overlap_poly.area,
                        })

        intersections = gpd.GeoDataFrame(records, geometry="overlap_geom", crs=self.gdf.crs)

        threshold = self.threshold
        too_short = intersections["shared_length_m"] < threshold
        is_consecutive = intersections["id_2"] == intersections["id_1"] + 1
        to_drop = is_consecutive & too_short
        intersections = intersections[~to_drop]

        return intersections


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument("--yml", "-y", required=True, help="Path to the configuration file")
    args = parser.parse_args()

    mls = MLSPipeline(config_path=args.yml)
    mls.pipeline()

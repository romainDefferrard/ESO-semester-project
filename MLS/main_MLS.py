import geopandas as gpd
import warnings
import argparse
import yaml
import sys

from PyQt6.QtWidgets import QApplication
from Utils.GUI_MLS import GUI_MLS

# Configuration
parser = argparse.ArgumentParser()
parser.add_argument("--yml", "-y", required=True, help="Path to the configuration file")
args = parser.parse_args()
config = yaml.safe_load(open(args.yml, "r"))

class MLS:
    def __init__(self, config):
        self.config = config
        self.filename = config["SHP_PATH"]
        self.epsg = config["EPSG"]
        self.buffer = config["BUFFER"]
        self.output_path = config["OUTPUT_PATH"]
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    def pipeline(self):
        self.open_file()
        self.add_buffer()
        self.intersections = self.get_intersections()
        self.run_gui()

    def run_gui(self):
        app = QApplication(sys.argv)
        window = GUI_MLS(gdf=self.gdf, 
                         intersections=self.intersections, 
                         output=self.output_path,
                        )
        window.show()
        app.exec()

    def open_file(self):
        self.gdf = gpd.read_file(self.filename)
        self.gdf = self.gdf.to_crs(epsg=self.epsg)

    def add_buffer(self):
        self.gdf["buffer"] = self.gdf.geometry.buffer(10)

    def get_intersections(self):
        records = []
        for i in range(len(self.gdf)):
            for j in range(i + 1, len(self.gdf)):
                buf1 = self.gdf.loc[i, "buffer"]
                buf2 = self.gdf.loc[j, "buffer"]

                if buf1.intersects(buf2):
                    overlap_poly = buf1.intersection(buf2)

                    if not overlap_poly.is_empty:
                        # Intersect original lines with the overlap polygon
                        line1 = self.gdf.loc[i, "geometry"].intersection(overlap_poly)
                        line2 = self.gdf.loc[j, "geometry"].intersection(overlap_poly)

                        # Take union of both clipped lines (they may be on opposite sides of the polygon)
                        shared_line = line1.union(line2)

                        # Get total length of shared line(s)
                        shared_length = shared_line.length if not shared_line.is_empty else 0

                        records.append(
                            {
                                "id_1": self.gdf.loc[i, "id"],
                                "id_2": self.gdf.loc[j, "id"],
                                "overlap_geom": overlap_poly,
                                "shared_length_m": shared_length,
                                "overlap_area_m2": overlap_poly.area,
                            }
                        )
        intersections = gpd.GeoDataFrame(records, geometry="overlap_geom", crs=self.gdf.crs)

        return intersections


   
if __name__ == "__main__":

    mls = MLS(config)
    mls.pipeline()

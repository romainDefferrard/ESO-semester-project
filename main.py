from Code.flight_data import FlightData
from Code.raster_loader import RasterLoader
from Code.footprint_generator import Footprint
from Code.patch_generator import PatchGenerator

import os
import numpy as np

# Load flight data and intervals
GPSTIME_PATH = "Config/GPSTime_config.json"
FLIGHT_PATH = "Data/Vallet2020_CH1903.txt"
# Load Raster file 
# A MODIF, moyen de choisir la projection??
MNT_PATH = "Data/espg2056_raster.tif"


# Flight loader
flight_manager = FlightData(GPSTIME_PATH, FLIGHT_PATH)
flight_bounds = flight_manager.bounds
flights = flight_manager.flights

# Raster loader 
raster_basemap = RasterLoader(MNT_PATH, epsg=2056, flight_bounds=flight_bounds)
raster = raster_basemap.raster
x_mesh, y_mesh = raster_basemap.x_mesh, raster_basemap.y_mesh 
raster_mesh = (x_mesh, y_mesh)

# Compute fligths footprint and superposition zone
footprint = Footprint(raster=raster, raster_mesh=raster_mesh, flights=flights)

# mask of superpos for flights 1 & 2
superposition = footprint.superpos_masks[0]

patch_gen = PatchGenerator(superpos_zones=superposition, raster_mesh=raster_mesh, raster=raster)
superpos_contour = patch_gen.contour_coords
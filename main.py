from Code.flight_data import FlightData
from Code.raster_loader import RasterLoader
from Code.footprint_generator import Footprint
from Code.patch_generator import PatchGenerator
from Code.las_extractor import LasExtractor

import os
import numpy as np
import time

# Load flight data and intervals
GPSTIME_PATH = "Config/GPSTime_config.json"
FLIGHT_PATH = "Data/Vallet2020_CH1903.txt"

# Load Raster file 
# A MODIF, moyen de choisir la projection??
MNT_PATH = "Data/espg2056_raster.tif"

# Dataset name, for output directory purpose
DATASET_NAME = 'DataTesting'

def main():     
        
    # Flight loader
    flight_manager = FlightData(GPSTIME_PATH, FLIGHT_PATH)
    flight_bounds = flight_manager.bounds
    flights = flight_manager.flights

    # Raster loader 
    raster_basemap = RasterLoader(MNT_PATH, epsg=2056, flight_bounds=flight_bounds)
    raster = raster_basemap.raster
    x_mesh, y_mesh = raster_basemap.x_mesh, raster_basemap.y_mesh 
    raster_mesh = (x_mesh, y_mesh)

    # Compute flights footprint and superposition zones
    footprint = Footprint(raster=raster, raster_mesh=raster_mesh, flights=flights)

    # mask of superpos for flights 1 & 2
    superposition = footprint.superpos_masks[0]

    band_length = 50
    band_width = 100
    sample_distance = 100
    patch_params = (band_length, band_width, sample_distance)

    for idx, (flight_i, flight_j) in enumerate(footprint.superpos_flight_pairs):
        # Get superposition mask for this flight pair
        superposition = footprint.superpos_masks[idx]
        
        # Create directory for this flight pair
        pair_dir = f"Output/{DATASET_NAME}/Flights_{flight_i}_{flight_j}"
        os.makedirs(pair_dir, exist_ok=True)
        print(f"Processing superposition between flights {flight_i} and {flight_j}")

        # Generate patches for this superposition zone
        patch_gen = PatchGenerator(superpos_zones=superposition, 
                                raster_mesh=raster_mesh, 
                                raster=raster, 
                                patch_params=patch_params)
        patches_poly = patch_gen.patches

        # Creating the new .las files 
        for patch_idx, patch in enumerate(patches_poly):
            
            # Pas le truc le plus clean du monde... 3eme for loop
            for flight in [flight_i, flight_j]:
                input_file = f"Data/ALS{flight}_baseline.las"
                output_file = f"{pair_dir}/patch_{patch_idx}_flight_{flight}.las"

                print(f"Extracting patch {patch_idx} for flight_{flight}...")

                # Extract LAS data for this patch
                LasExtractor(input=input_file, output=output_file, patch=patch)
            
   
if __name__ == "__main__": 
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # Temps total pour les 16 patchs 126s = 2min
    # idee: méthode .within prend trop de temps et peut être que en regardant 
    # une bounding box avec les coordonnées ça serait plus rapide ???
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
from multiprocessing import Manager, Pool

import logging
import yaml
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
import laspy
import numpy as np


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

def load_data(pair_mode, lidar_scan_mode, lidar_tilt_angle):
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
    start_time = time.time()
    footprint = Footprint(raster=raster, raster_mesh=raster_mesh, flights=flights, mode=pair_mode, lidar_scan_mode=lidar_scan_mode, lidar_tilt_angle=lidar_tilt_angle)
    logging.info(f"footprint terminée en {time.time() - start_time:.2f} secondes.")

    
    return raster_loader, footprint

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

def create_tasks_las_reading(footprint, las_pattern):
    """Generate tasks for reading .laz files using multiprocessing."""
    tasks = []
    flight_ids = set()  # To track unique flights

    # Iterate over each flight pair in the footprint superposition zones
    for idx, (flight_i, flight_j) in enumerate(footprint.superpos_flight_pairs):
        flight_ids.add(flight_i)
        flight_ids.add(flight_j)

    logging.info(f"Unique flights to read: {len(flight_ids)}")

    # For each unique flight, create a task to read the corresponding .laz file
    for flight_id in flight_ids:
        input_file = las_pattern.format(flight_id=flight_id)  # Format the file name for this flight
        tasks.append((input_file,))  # Make sure each task is a single-element tuple

    return tasks

def extract_patch(flight_id, patch_idx, patch, pair_dir, las_pattern):
    """Extract LAS data for a given flight and patch."""
    input_file = las_pattern.format(flight_id=flight_id)  # Use the pattern from the config
    output_file = f"{pair_dir}/patch_{patch_idx}_flight_{flight_id}.laz"
    flight_pair = os.path.basename(pair_dir)
    
    # Retrieve the correct LAS data and coordinates using the dictionaries
    
    extractor = LasExtractor(input=input_file, patch=patch)
    extractor.patch = patch # add patch to the extractor

    logging.info(f"Extracting patch {patch_idx} for flight {flight_id} ({flight_pair})")
    #extractor = LasExtractor(input=input_file, output=output_file, patch=patch)
    extractor.process_patch()
    logging.info(f"Extraction completed for patch {patch_idx} in flight {flight_id}. Output saved to {output_file}")

def read_laz(task):
    input_file = task
    logging.info(f"Reading {input_file}")
    extractor = LasExtractor(input_file, patch=None)
    extractor.laz_read()
    logging.info(f"Finished reading {input_file}")
    
def run_extraction(tasks_patch, tasks_laz):
    """Extraction of patches in corresponding flights .las"""

    start_time = time.time()

    # First, read all .laz files and store coordinates in result_dict
    extractor = LasExtractor(input=None, patch=None)
        
    with Pool() as pool:
        pool.starmap(read_laz, tasks_laz)

    # Now you have all results in a single dictionary
    # You can access las_data and coords like:

    # Now, run extraction using the coordinates stored in result_dict
    logging.info(f"Starting LAS extraction for {len(tasks_patch)} patches...")

    with Pool() as pool:
        pool.starmap(extract_patch, tasks_patch)
    logging.info(f"Extraction completed in {time.time() - start_time:.2f} seconds.")

def run_extraction3(footprint, patches_poly, las_format, output_format):
    """ On test de faire un jeu de donnée à la fois et d'intégrer le multiprocessing 
        directement dans la classe comme ça on perd pas les données de l'extraction
        et surtout on la fait pas en boucle inutilement (!)
    """
    start_time = time.time()

    # Liste des vols uniques
    flight_ids = set(flight_i for flight_i, _ in footprint.superpos_flight_pairs) | \
                 set(flight_j for _, flight_j in footprint.superpos_flight_pairs)

    for flight_id in flight_ids:
        logging.info(f"Traitement du vol {flight_id}...")

        # Charger le fichier .laz (séquentiel)
        input_file = las_format.format(flight_id=flight_id)  # Adapter selon ton naming
        extractor = LasExtractor(input_file, patches=None)

        if not extractor.laz_read():
            logging.error(f"Impossible de lire {input_file}, passage au vol suivant.")
            continue

        # Trouver les patches correspondants à ce vol
        patches_for_flight = []
        for idx, (flight_i, flight_j) in enumerate(footprint.superpos_flight_pairs):
            if flight_i == flight_id or flight_j == flight_id:
                patches_for_flight.extend(patches_poly[idx])
        # Lancer l'extraction en parallèle
        extractor.process_all_patches(patches_for_flight, OUTPUT_DIR, )

    logging.info(f"Extraction terminée en {time.time() - start_time:.2f} secondes.")

def run_extraction2(footprint, patches_poly, LAS_DIR, OUTPUT_DIR):
    """Exécute l'extraction des patches pour chaque paire de vols et sauvegarde les résultats dans des dossiers distincts."""
    logging.info("Début du processus d'extraction...")
    start_time = time.time()
    
    for idx, (flight_i, flight_j) in enumerate(footprint.superpos_flight_pairs):
        flight_patch = patches_poly[idx]
        
        # Création du répertoire spécifique à la paire de vols
        pair_dir = f"{OUTPUT_DIR}/Flights_{flight_i}_{flight_j}"
        os.makedirs(pair_dir, exist_ok=True)

        # Extraction pour le premier vol de la paire
        logging.info(f"Traitement du vol {flight_i}...")
        input_file = LAS_DIR.format(flight_id=flight_i)
        extractor_i = LasExtractor(input_file, flight_patch)

        if extractor_i.laz_read():
            extractor_i.process_all_patches(flight_patch, OUTPUT_DIR, flight_i, pair_dir)

        # Extraction pour le deuxième vol de la paire
        logging.info(f"Traitement du vol {flight_j}...")
        input_file = LAS_DIR.format(flight_id=flight_j)
        extractor_j = LasExtractor(input_file, flight_patch)

        if extractor_j.laz_read():
            extractor_j.process_all_patches(flight_patch, OUTPUT_DIR, flight_j, pair_dir)
    logging.info(f"Extraction terminée en {time.time() - start_time:.2f} secondes.")

    logging.info("Extraction complète pour toutes les paires de vols.")

def main():
    pair_mode = "successive"  # Set mode: "successive" or "all" flight pairs 
    
    raster_loader, footprint = load_data(pair_mode=pair_mode, 
                                         lidar_scan_mode= 'right', 
                                         lidar_tilt_angle=15)
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
    #    tasks_patch = create_tasks(footprint, new_patches_poly)
    #    tasks_laz = create_tasks_las_reading(footprint, LAS_DIR)

    #    run_extraction(tasks_patch, tasks_laz)
        run_extraction2(footprint, new_patches_poly, LAS_DIR, OUTPUT_DIR)

    else:
        logging.info("Window closed without extraction.")



if __name__ == "__main__":
    main()

"""Notes de modif à faire: 
- modifier mode de centerline parmis une liste (voir résultats du val d'Arpette)

- effecer les patches sans correspondance dans l'autre vol (si erreur à l'extraction)


Updates: 
- enorme galère mercredi, j'ai changé la pipeline apres avoir passer la journée à charcher le problème, avec des taille de donnée comme ça le multiprocessing 
ne marchait pas pour faire des taches aussi "petites" (extraction)
- aussi un peu des effets de bord imprevisible -> faire contition pour annuler la paire de patch si un des 
deux n'a pas marché

- nouvelle version du get_footprint prend en compte une FOV corrigée en fonction de l'angle du lidar
    - mais je fonctionne toujours pareil ou je regarde à 360° en dessous de l'appareil (dans un cone)
    - j'aimerais faire un scanning réel àavec un certain angle pour que globalement le debut et la fin soient 
    - plus réaliste car la je coupe à 90° de la trajectoire alors qu'en réalité je devrait couper à un certain angle 

- Deux modes footprint
    - classique juste angle FOV
    - avec un scan à un certain angle 
    - lequel le mieux ??
    
    
Centerline:
- Il faut que je vois encore d'autres methodes que PCA pour voir si j'ai des meilleurs résultats 
- est ce que tu veux un truc parallele aux vols ou on s'en fout?
- le problème c'est que une fois que j'ai corrigé la manière que je faisais mes footprint 
les résultats de la centerline sont devenus beaucoup moins on

Notes tests: 
- patch 100m toute la longueur 
    -320s k=10
    -273s k=5 
    
Retour Aurel:
- sérieux effets de bord au début et fin des vols puisque le lidar je vise pas droit mais oblique 
- FOV du capteur ?
- multiprocessing marchait plus -> plus sur l'extraction mais sur générer les footprint (assez efficace sur l'extraction)
    - essayer de l'intégrer différement sur l'extraction à l'avenir 
- méthode la plus réaliste de footprint donne mauvais résultat PCA

"""
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
import logging
import yaml
import argparse

# Configuration
parser = argparse.ArgumentParser()
parser.add_argument("--yml", "-y", required=True, help="Path to the configuration file")
args = parser.parse_args()

# Config paths
config = yaml.safe_load(open(args.yml, "r"))
TRAJECTORY_PATH = config["TRAJECTORY_PATH"]
MNT_PATH = config["MNT_PATH"]
OUTPUT_DIR = config["OUTPUT_DIR"]
LAS_DIR = config["LAS_DIR"]
LOG_DIR = config["LOG_DIR"]
DATASET_NAME = config["DATASET_NAME"]
LIDAR_SCAN_MODE = config["LIDAR_SCAN_MODE"]
LIDAR_TILT_ANGLE = config["LIDAR_TILT_ANGLE"]
LIDAR_FOV = config["LIDAR_FOV"]
PAIR_MODE = config["PAIR_MODE"]
FLIGHT_SAMPLING = config["FLIGHT_SAMPLING"]

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

def load_data():
    # load flights, raster and compute footprints
    logging.info("Loading flight and raster data...")

    # Load flight data
    flight_manager = FlightData(LAS_DIR, LOG_DIR, TRAJECTORY_PATH, DATASET_NAME)
    flight_bounds = flight_manager.bounds
    flights = flight_manager.flights

    # Load raster data
    raster_loader = RasterLoader(MNT_PATH, epsg=2056, flight_bounds=flight_bounds)
    raster = raster_loader.raster
    raster_mesh = (raster_loader.x_mesh, raster_loader.y_mesh)
    time0 = time.time()
    # Compute footprint
    footprint = Footprint(
        raster=raster,
        raster_mesh=raster_mesh,
        flights=flights,
        mode=PAIR_MODE,
        lidar_scan_mode=LIDAR_SCAN_MODE,
        lidar_tilt_angle=LIDAR_TILT_ANGLE,
        fov=LIDAR_FOV, 
        sampling_interval=FLIGHT_SAMPLING
    )
    logging.info(f"footprint: {time.time() - time0:.2f}s")
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
        flight_pairs=footprint.superpos_flight_pairs,
        output_dir = OUTPUT_DIR
    )
    window.show()
    app.exec()

    return window.control_panel.extraction_state, window.control_panel.new_patches_poly

def run_extraction(footprint, patches_poly, LAS_DIR, OUTPUT_DIR):
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

        if extractor_i.read_point_cloud():
            extractor_i.process_all_patches(flight_patch, OUTPUT_DIR, flight_i, pair_dir)

        # Extraction pour le deuxième vol de la paire
        logging.info(f"Traitement du vol {flight_j}...")
        input_file = LAS_DIR.format(flight_id=flight_j)
        extractor_j = LasExtractor(input_file, flight_patch)

        if extractor_j.read_point_cloud():
            extractor_j.process_all_patches(flight_patch, OUTPUT_DIR, flight_j, pair_dir)
    logging.info(f"Extraction terminée en {time.time() - start_time:.2f} secondes.")

    logging.info("Extraction complète pour toutes les paires de vols.")

def main():

    raster_loader, footprint = load_data()
    raster, raster_mesh = raster_loader.raster, (raster_loader.x_mesh, raster_loader.y_mesh)

    # Create PatchGenerator instance with all superposition zones
    patch_gen = PatchGenerator(superpos_zones=footprint.superpos_masks, raster_mesh=raster_mesh, raster=raster, patch_params=(50, 100, 400))
    patches_all = patch_gen.patches_list
    centerlines_all = patch_gen.centerlines_list  # All centerlines
    contours_all = patch_gen.contours_list  # All contours

    # Run GUI with all patches, centerlines, and contours
    extraction_state, new_patches_poly = run_gui(footprint, patches_all, centerlines_all, contours_all, raster, raster_mesh)

    if extraction_state:
        run_extraction(footprint, new_patches_poly, LAS_DIR, OUTPUT_DIR)
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


""" Footprint time analysis 
        
        def main():
    
    def compare_masks(reference_masks, test_masks):
        if len(reference_masks) != len(test_masks):
            return False  # Nombre de masques différent

        for ref, test in zip(reference_masks, test_masks):
            if not np.array_equal(ref, test):
                return False  # Différence détectée

        return True  # Les masques sont identiques

    sampling_values = [1,2,3,4,5,10,20,30, 50,70, 100,]  # Différents niveaux d'échantillonnage
    times = []
    masks_sampling = []
    _, footprint, elapsed_time = load_data(1)
    reference_masks = footprint.superpos_masks

    for sampling in sampling_values:
        raster_loader, footprint, elapsed_time = load_data(sampling)
        masks = footprint.superpos_masks
        masks_sampling.append(masks)
        print(f"Sampling Interval: {sampling} | Temps: {elapsed_time:.2f} sec")
        identical = compare_masks(reference_masks, masks)
        print(f"Sampling {sampling} → Identique au raster de référence ? {'✅ Oui' if identical else '❌ Non'}")
        times.append(elapsed_time)
        
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(sampling_values, times, marker='o', linestyle='-')

    plt.xlabel("Distance between samples (number of steps)")
    plt.ylabel("Elapsed time (s)")
    plt.title("Footprint time analysis")
    #plt.xscale("log")  # Car l'échantillonnage varie de manière exponentielle
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.savefig("sampling_vs_time.png", dpi=300)
    plt.show()

    def plot_mask_differences(reference_masks, test_masks, sampling_values):
        
        num_samples = len(test_masks)
        fig, axes = plt.subplots(num_samples, 3, figsize=(5, 3 * num_samples))

        if num_samples == 1:
            axes = [axes]  # S'assurer que axes est toujours une liste

        for i, (sampling, test_mask) in enumerate(zip(sampling_values, test_masks)):
            ref_mask = np.array(reference_masks).squeeze()  # Convertir en array et supprimer les dimensions inutiles
            test_mask = np.array(test_mask).squeeze()
            diff_mask = np.logical_xor(ref_mask, test_mask)  # Met en évidence les différences

            # Affichage des masques
            axes[i][0].imshow(ref_mask, cmap="gray")
            axes[i][0].set_title("Référence")

            axes[i][1].imshow(test_mask, cmap="gray")
            axes[i][1].set_title(f"Sampling {sampling}")

            axes[i][2].imshow(diff_mask, cmap="hot")
            axes[i][2].set_title("Différences")

            for ax in axes[i]:
                ax.axis("off")

        plt.tight_layout()
        plt.savefig('sampling4.png', dpi=300)
        plt.show()

    # Exemple d'utilisation
    # Supposons que reference_masks soit un tableau numpy binaire de référence
    # Et test_masks soit une liste des masques obtenus avec différents niveaux de sampling
    plot_mask_differences(reference_masks, masks_sampling, sampling_values)
    exit()

    raster_loader, footprint = load_data(sampling_interval)
    raster, raster_mesh = raster_loader.raster, (raster_loader.x_mesh, raster_loader.y_mesh)

    # Create PatchGenerator instance with all superposition zones
    patch_gen = PatchGenerator(superpos_zones=footprint.superpos_masks, raster_mesh=raster_mesh, raster=raster, patch_params=(50, 100, 400))

    # Now, patch_gen contains all the patches and centerlines
    patches_all = patch_gen.patches_list
    centerlines_all = patch_gen.centerlines_list  # All centerlines
    contours_all = patch_gen.contours_list  # All contours

    # Run GUI with all patches, centerlines, and contours
    extraction_state, new_patches_poly = run_gui(footprint, patches_all, centerlines_all, contours_all, raster, raster_mesh)

    if extraction_state:
        run_extraction(footprint, new_patches_poly, LAS_DIR, OUTPUT_DIR)
    else:
        logging.info("Window closed without extraction.")

"""
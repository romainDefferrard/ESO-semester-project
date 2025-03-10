import laspy 
import numpy as np 
from shapely.geometry import Point
from sklearn.neighbors import KNeighborsClassifier
from multiprocessing import Pool
import pandas as pd 
import os
import copy
import logging
import mapping


#logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

class LasExtractor():
    def __init__(self, input, patches):
        self.input_file = input
        self.patches = patches # list of patches
                
        self.las = None  # To store the LAS data (useful for the header)
        self.coords = None # and the coordinates x,y
        
        self.coords_mask = None
        
    def laz_read(self):
        """LAS/LAZ reader depending on the format
        """
        if not os.path.exists(self.input_file):
            logging.error(f"File not found: {self.input_file}")
            return False

        with laspy.open(self.input_file) as fh:
            self.las = fh.read()  # Read the file fully
            self.coords = np.vstack((self.las.x, self.las.y)).transpose()  # Extract XYZ
        
        return True 
    

    def patch_filtering_knn_classifier(self, patch, k=5, prob_threshold=0.3, sample_factor=0.1):
        """Filtrage des points dans un patch avec KNN"""
        
        min_x, min_y, max_x, max_y = patch.bounds
        bbox_mask = ((self.coords[:, 0] >= min_x) & (self.coords[:, 0] <= max_x) & 
                    (self.coords[:, 1] >= min_y) & (self.coords[:, 1] <= max_y))
        points_in_bbox = self.coords[bbox_mask]

        # Vérifier s'il y a des points dans la bounding box
        if len(points_in_bbox) == 0:
            logging.warning(f"Aucun point filtré dans le patch {patch}. Fichier non sauvegardé.")
            return None, None

        df = pd.DataFrame(points_in_bbox, columns=['x', 'y'])
        # Sélection aléatoire d'un sous-ensemble de points
        sample_size = int(sample_factor * len(df))  # Évite de prendre 0 échantillons
        df_short = df.sample(n=sample_size, random_state=42).copy()

        # Labellisation des points (1 si à l'intérieur, 0 sinon)
        df_short['labels'] = [patch.contains(Point(px, py)) for px, py in zip(df_short.x, df_short.y)]
 

        # Entraînement du KNN
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(df_short[['x', 'y']], df_short['labels'])
        
        # Prédiction de probabilité pour tous les points du patch
        df['predict'] = knn.predict_proba(df[['x', 'y']])[:, 1]

        # Création du masque filtré
        mask = df['predict'].values > prob_threshold
        self.coords_mask = np.zeros(len(self.coords), dtype=bool)
        self.coords_mask[bbox_mask] = mask
        
        return bbox_mask, mask
        
        
    def copy_header(self):
        self.header = copy.deepcopy(self.las.header)
        filtered_point_count = np.sum(self.coords_mask)
        self.header.point_count = filtered_point_count  
        

            
    def extract_patch(self, patch, output_dir, flight_id, pair_dir, patch_idx):
        """Extrait un patch spécifique"""
        output_file = f"{output_dir}/patch_{patch_idx}.laz"
        output_file = f"{pair_dir}/patch_{patch_idx}_flight_{flight_id}.laz"

        
        logging.info(f"Démarrage de l'extraction du patch {patch_idx} pour le vol {flight_id}.")
        bbox_mask, mask = self.patch_filtering_knn_classifier(patch, k=5, prob_threshold=0.3, sample_factor=0.2)
        
        if bbox_mask is None or mask is None or np.sum(mask) == 0:
            logging.warning(f"Aucun point filtré dans le patch {patch_idx}, fichier non sauvegardé.")
            return
        
        self.copy_header()

        new_las = laspy.LasData(self.header)
        for dimension in self.las.point_format.dimension_names:
            data = getattr(self.las, dimension)
            setattr(new_las, dimension, data[self.coords_mask])

        new_las.write(output_file)
        logging.info(f"Patch: {patch_idx} , vol: {flight_id}, output: {output_file} enregistré")

        
    def process_all_patches(self, patches, output_dir, flight_id, pair_dir):
        """Exécute l'extraction des patches en parallèle"""
        logging.info(f"Extraction parallèle pour {len(patches)} patches.")
        
        for idx, patch in enumerate(patches):
            self.extract_patch(patch, output_dir, flight_id, pair_dir, idx)
        #with Pool() as pool:
        #    pool.starmap(self.extract_patch, [(patch, output_dir) for patch in patches])

        logging.info(f"Extraction terminée pour le vol {flight_id}")
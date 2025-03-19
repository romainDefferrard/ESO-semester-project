import laspy
import numpy as np
from shapely.geometry import Point
from sklearn.neighbors import KNeighborsClassifier
from multiprocessing import Pool
import pandas as pd
import os
import copy
import logging


# logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


class LasExtractor:
    def __init__(self, input, patches):
        self.input_file = input
        self.patches = patches  # list of patches

        self.las = None  # To store the LAS data (useful for the header)
        self.coords = None  # and the coordinates x,y

        # useful in ascii format
        self.z = None
        self.gps_times = None  # GPS time
        self.intensities = None  # Intensity or Classification

        self.coords_mask = None

        self.file_format = self.detect_file_format()

    def detect_file_format(self):
        if self.input_file.endswith(".laz") or self.input_file.endswith(".las"):
            return "laz"
        elif self.input_file.endswith(".TXYZS") or self.input_file.endswith(".txt"):
            return "TXYZS"
        else:
            raise ValueError(f"Unsupported file format: {self.input_file}. Supported: .laz, .las, .TXYZS")

    def read_point_cloud(self):
        if self.file_format == "laz":
            return self.las_read()
        elif self.file_format == "TXYZS":
            return self.ascii_read()

    def las_read(self):
        """LAS/LAZ reader depending on the format"""
        if not os.path.exists(self.input_file):
            logging.error(f"File not found: {self.input_file}")
            return False

        with laspy.open(self.input_file) as fh:
            self.las = fh.read()  # Read the file fully
            self.coords = np.vstack((self.las.x, self.las.y)).transpose()  # Extract XYZ

        return True

    def ascii_read(self):
        """ASCII reader"""
        if not os.path.exists(self.input_file):
            logging.error(f"File not found: {self.input_file}")
            return False

        df = pd.read_csv(self.input_file, delimiter="\t", header=None, dtype=float)

        if df.shape[1] < 7:
            raise ValueError("File does not contain enough columns. Expected at least 7 (T, X, Y, Z, S1, S2, S3)")

        self.gps_times = df.iloc[:, 0].values  # Time
        self.coords = df.iloc[:, 1:3].values  # X, Y
        self.z = df.iloc[:, 3].values  # Z
        self.intensities = df.iloc[:, 4:7].values  # Intensity values (3 columns)

        return True

    def patch_filtering_knn_classifier(self, patch, k=5, prob_threshold=0.3, sample_factor=0.1):
        """Filtrage des points dans un patch avec KNN"""

        min_x, min_y, max_x, max_y = patch.bounds
        bbox_mask = (self.coords[:, 0] >= min_x) & (self.coords[:, 0] <= max_x) & (self.coords[:, 1] >= min_y) & (self.coords[:, 1] <= max_y)
        points_in_bbox = self.coords[bbox_mask]

        # Vérifier s'il y a des points dans la bounding box
        if len(points_in_bbox) == 0:
            logging.warning(f"Aucun point filtré dans le patch {patch}. Fichier non sauvegardé.")
            return None, None

        df = pd.DataFrame(points_in_bbox, columns=["x", "y"])
        # Sélection aléatoire d'un sous-ensemble de points
        sample_size = int(sample_factor * len(df))  # Évite de prendre 0 échantillons
        df_short = df.sample(n=sample_size, random_state=42).copy()

        # Labellisation des points (1 si à l'intérieur, 0 sinon)
        df_short["labels"] = [patch.contains(Point(px, py)) for px, py in zip(df_short.x, df_short.y)]

        # Entraînement du KNN
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(df_short[["x", "y"]], df_short["labels"])

        # Prédiction de probabilité pour tous les points du patch
        df["predict"] = knn.predict_proba(df[["x", "y"]])[:, 1]

        # Création du masque filtré
        mask = df["predict"].values > prob_threshold
        self.coords_mask = np.zeros(len(self.coords), dtype=bool)
        self.coords_mask[bbox_mask] = mask

        return bbox_mask, mask

    def write_las(self, output_file):

        self.copy_header()
        new_las = laspy.LasData(self.header)
        for dimension in self.las.point_format.dimension_names:
            data = getattr(self.las, dimension)
            setattr(new_las, dimension, data[self.coords_mask])

        new_las.write(output_file)
        logging.info(f"Saved extracted patch to {output_file}")

    def write_ascii(self, output_file):
        """Writes extracted points to .TXYZS ASCII format"""
        extracted_points = np.column_stack(
            (
                self.gps_times[self.coords_mask],
                self.coords[self.coords_mask, 0],
                self.coords[self.coords_mask, 1],
                self.z[self.coords_mask],
                self.intensities[self.coords_mask, 0],
                self.intensities[self.coords_mask, 1],
                self.intensities[self.coords_mask, 2],
            )
        )

        np.savetxt(output_file, extracted_points, delimiter="\t")
        logging.info(f"Saved extracted patch to {output_file}")

    def copy_header(self):
        self.header = copy.deepcopy(self.las.header)
        self.header.point_count = np.sum(self.coords_mask)

    def extract_patch(self, patch, output_dir, flight_id, pair_dir, patch_idx):
        """Extrait un patch spécifique"""
        output_file = f"{pair_dir}/patch_{patch_idx}_flight_{flight_id}.{self.file_format}"
        logging.info(f"Démarrage de l'extraction du patch {patch_idx} pour le vol {flight_id}.")

        bbox_mask, mask = self.patch_filtering_knn_classifier(patch, k=5, prob_threshold=0.3, sample_factor=0.2)

        if bbox_mask is None or mask is None or np.sum(mask) == 0:
            logging.warning(f"Aucun point filtré dans le patch {patch_idx}, fichier non sauvegardé.")
            return

        if self.file_format == "TXYZS":
            self.write_ascii(output_file)
        else:
            self.write_las(output_file)

    def process_all_patches(self, patches, output_dir, flight_id, pair_dir):
        """Exécute l'extraction des patches en parallèle"""

        for idx, patch in enumerate(patches):
            self.extract_patch(patch, output_dir, flight_id, pair_dir, idx)

        logging.info(f"Extraction terminée pour le vol {flight_id}")

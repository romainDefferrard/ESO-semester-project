import laspy
import numpy as np
from shapely.geometry import Point
from sklearn.neighbors import KNeighborsClassifier
from multiprocessing import Pool
import pandas as pd
import os
import copy
import logging
from laspy import ExtraBytesParams
from .patch_model import Patch
from typing import List, Tuple
import time
from matplotlib.path import Path
from .timer_logger import TimerLogger
import copy

from tqdm import tqdm

class LasExtractor:
    def __init__(self, config: dict, input_file: str, patches: List[Patch]):
        self.extraction_mode = config["EXTRACTION_MODE"]
        self.write_extra_file = config["WRITE_PATCH_FILE"]
        self.input_file = input_file
        self.patches = patches  # list of patches instances

        self.las = None  # To store the LAS data (useful for the header)
        self.coords = None  # and the coordinates x,y

        # useful in ascii format
        self.z = None
        self.gps_times = None  # GPS time
        self.intensities = None  # Intensity or Classification

        self.coords_mask = None
        self.timer = TimerLogger()


        self.file_format = self.detect_file_format()
        

    def detect_file_format(self) -> str:
        if self.input_file.endswith(".laz") or self.input_file.endswith(".las"):
            return "laz"
        elif self.input_file.endswith(".TXYZS") or self.input_file.endswith(".txt"):
            return "TXYZS"
        else:
            raise ValueError(f"Unsupported file format: {self.input_file}. Supported: .laz, .las, .TXYZS")

    def read_point_cloud(self) -> bool:
        if self.file_format == "laz":
            return self.las_read()
        elif self.file_format == "TXYZS":
            return self.ascii_read()

    def las_read(self) -> bool:
        """LAS/LAZ reader with timing."""
        if not os.path.exists(self.input_file):
            logging.error(f"File not found: {self.input_file}")
            return False

        #self.timer.start(f"Read LAS {os.path.basename(self.input_file)}")
        with laspy.open(self.input_file) as fh:
            self.las = fh.read()
            self.coords = np.vstack((self.las.x, self.las.y)).T
        #self.timer.stop(f"Read LAS {os.path.basename(self.input_file)}")

        return True


    def ascii_read(self) -> bool:
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

    def patch_filtering_knn_classifier(
        self, patch: Patch, k: int = 5, prob_threshold: float = 0.3, sample_factor: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
        polygon = patch.shapely_polygon
        min_x, min_y, max_x, max_y = polygon.bounds

        bbox_mask = (self.coords[:, 0] >= min_x) & (self.coords[:, 0] <= max_x) & (self.coords[:, 1] >= min_y) & (self.coords[:, 1] <= max_y)
        points_in_bbox = self.coords[bbox_mask]

        if len(points_in_bbox) == 0:
            return None, None

        df = pd.DataFrame(points_in_bbox, columns=["x", "y"])
        sample_size = max(1, int(sample_factor * len(df)))
        df_short = df.sample(n=sample_size, random_state=42).copy()

        df_short["labels"] = [polygon.contains(Point(px, py)) for px, py in zip(df_short.x, df_short.y)]

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(df_short[["x", "y"]], df_short["labels"])

        df["predict"] = knn.predict_proba(df[["x", "y"]])[:, 1]

        mask = df["predict"].values > prob_threshold
        self.coords_mask = np.zeros(len(self.coords), dtype=bool)
        self.coords_mask[bbox_mask] = mask

        return bbox_mask, mask

    def write_las(self, output_file: str) -> None:

        self.copy_header()
        new_las = laspy.LasData(self.header)
        for dimension in self.las.point_format.dimension_names:
            data = getattr(self.las, dimension)
            setattr(new_las, dimension, data[self.coords_mask])

        new_las.write(output_file)

    def write_ascii(self, output_file: str) -> None:
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

    def copy_header(self) -> None:
        self.header = copy.deepcopy(self.las.header)
        self.header.point_count = np.sum(self.coords_mask)

    def extract_patch(self, patch: Patch, flight_id: str, pair_dir: str):
        """Extracts a single patch"""
        output_file = f"{pair_dir}/patch_{patch.id}_flight_{flight_id}.{self.file_format}"
        bbox_mask, mask = self.patch_filtering_knn_classifier(patch)
        if bbox_mask is None or mask is None or np.sum(mask) == 0:
            logging.warning(f"No filtered points in patch {patch.patch_id}, skipping save.")
            return

        if self.file_format == "TXYZS":
            self.write_ascii(output_file)
        else:
            self.write_las(output_file)


    def process_all_patches(self, patches: List[Patch], output_dir: str, flight_id: str, pair_dir: str) -> None:
        if self.extraction_mode == "independent":
            for patch in patches:
                os.makedirs(pair_dir, exist_ok=True)
                self.extract_patch(patch, flight_id, pair_dir)
        elif self.extraction_mode == "Extra_Bytes":
            self.encode_patches_extrabytes(patches, flight_id, output_dir)
        elif self.extraction_mode == "binary": 
            self.encode_patches_binary(patches, flight_id, output_dir)
        else:
            raise ValueError(f"Unknown extraction mode: {self.extraction_mode}")
        
        
    def encode_patches_extrabytes(self, patches: List[Patch], flight_id: str, output_dir: str) -> None:
        """
        Crée ou réinitialise les extrabytes dans self.las  + 
        les écrit dans un fichier externe .patch si activé.
        Ecrase le fichier d'entrée avec le nouveau contenu.
        """
        filename = os.path.basename(self.input_file)
        output_laz = os.path.join(output_dir, filename)
        output_patch_file = f"{output_dir}/flight_{flight_id}.patch"
        
        new_las = self.las

        self.timer.start("Add or reset extra dimensions")
        self.safe_add_or_reset("num_patches", np.uint8, new_las)
        self.safe_add_or_reset("patch_ids_1", np.uint8, new_las)

        created_fields = {"patch_ids_1"}
        field_data = {"num_patches": new_las["num_patches"],
                    "patch_ids_1": new_las["patch_ids_1"]}
        self.timer.stop("Add or reset extra dimensions")

        self.timer.start(f"Masking flight {flight_id}")
        patch_masks = []
        for patch in patches:
            selected_indices = self.fast_patch_mask(patch)
            if len(selected_indices) > 0:
                patch_masks.append((patch, selected_indices))
        self.timer.stop(f"Masking flight {flight_id}")

        self.timer.start(f"ExtraBytes writing {flight_id}")
        for patch, selected_indices in patch_masks:
            levels = new_las.num_patches[selected_indices].copy()
            new_las.num_patches[selected_indices] += 1

            for level in np.unique(levels):
                idxs = selected_indices[levels == level]
                field_name = f"patch_ids_{level + 1}"

                if field_name not in created_fields:
                    self.safe_add_or_reset(field_name, np.uint8, new_las)
                    field_data[field_name] = new_las[field_name]
                    created_fields.add(field_name)

                new_las[field_name][idxs] = patch.id
        self.timer.stop(f"ExtraBytes writing {flight_id}")

        self.timer.start("Writing new LAS in input file (safe overwrite)")
        new_las.write(output_laz)
        os.replace(output_laz, self.input_file)  
        self.timer.stop("Writing new LAS in input file (safe overwrite)")

        # écrire .patch uniquement pour les points dans au moins un patch
        if self.write_extra_file:
            self.timer.start(f"Write patch file flight {flight_id}")
            indices = np.where(new_las["num_patches"] > 0)[0]
            arrays = [new_las["num_patches"][indices]] + [
                field_data[key][indices] for key in sorted(field_data.keys()) if key not in {"num_patches"}
            ]
            extrabytes = np.column_stack([indices] + arrays)
            np.savetxt(output_patch_file, extrabytes, delimiter="\t", fmt="%d")
            self.timer.stop(f"Write patch file flight {flight_id}")

    def encode_patches_binary(self, patches: List[Patch], flight_id: str, output_dir: str):
        output_file = f"{output_dir}/flight_{flight_id}.patchbin"
        num_points = len(self.coords)

        patch_array = np.zeros((num_points, 2), dtype=np.uint8)
        
        #self.timer.start(f"Masking flight {flight_id}")
        patch_masks = []
        for patch in patches:
            selected_indices = self.fast_patch_mask(patch)
            if len(selected_indices) > 0:
                patch_masks.append((patch, selected_indices))
        #self.timer.stop(f"Masking flight {flight_id}")

        #self.timer.start("Filling phase")
        for patch, indices in patch_masks:
            levels = patch_array[indices, 0].copy()  # num_patches column
            patch_array[indices, 0] += 1

            for level in np.unique(levels):
                idxs = indices[levels == level]
                col = int(level) + 1  # patch_ids_1, patch_ids_2, etc.

                # If needed, add new column
                if col >= patch_array.shape[1]:
                    new_column = np.zeros((num_points, 1), dtype=np.uint8)
                    patch_array = np.hstack((patch_array, new_column))

                patch_array[idxs, col] = patch.id
        #self.timer.stop("Filling phase")

        #self.timer.start("Write binary file")
        num_columns = patch_array.shape[1]
        with open(output_file, "wb") as f:
            f.write(np.array([num_columns], dtype=np.uint8).tobytes())  # header = 1 byte
            f.write(patch_array.tobytes())  # full array
        #self.timer.stop("Write binary file")


    def encode_extrabytes_to_output(self, patches: List[Patch], flight_id: str, output_dir: str) -> None:
        """celle de base copie le .laz de l'input et écrit les extrabytes dans de dossier Output
        """
        output_file = f"{output_dir}/output_flight_{flight_id}.laz"
        num_points = len(self.coords)
        
        self.timer.start("Init new LAS")
        new_las = self.las
        
        new_las.add_extra_dim(ExtraBytesParams(name="num_patches", type=np.uint8))
        new_las.num_patches = np.zeros(num_points, dtype=np.uint8)
        
        new_las.add_extra_dim(ExtraBytesParams(name="patch_ids_1", type=np.uint8))
        new_las.patch_ids_1 = np.zeros(num_points, dtype=np.uint8)
        self.timer.stop("Init new LAS")
        created_fields = {"patch_ids_1"}
        field_data = {"patch_ids_1": new_las["patch_ids_1"]}

        self.timer.start(f"Masking flight {flight_id}")
        for patch in tqdm(patches, desc=f"Masking flight {flight_id}", unit="patch"):
            selected_indices = self.fast_patch_mask(patch)
            levels = new_las.num_patches[selected_indices].copy()
            new_las.num_patches[selected_indices] += 1

            for level in np.unique(levels):
                idxs = selected_indices[levels == level]
                field_name = f"patch_ids_{level + 1}"

                if field_name not in created_fields:
                    new_las.add_extra_dim(ExtraBytesParams(name=field_name, type=np.uint8))
                    new_las[field_name] = np.zeros(num_points, dtype=np.uint8)
                    field_data[field_name] = new_las[field_name]
                    created_fields.add(field_name)

                new_las[field_name][idxs] = patch.id
        self.timer.stop(f"Masking flight {flight_id}")

        self.timer.start(f"Write LAS flight {flight_id}")
        new_las.write(output_file)
        self.timer.stop(f"Write LAS flight {flight_id}")


    def safe_add_or_reset(self, name, dtype, las):
        """Overwrite existing extrabytes or create if missing."""
        if name in las.point_format.extra_dimension_names:
            las[name][:] = 0
        else:
            las.add_extra_dim(ExtraBytesParams(name=name, type=dtype))
            las[name] = np.zeros(len(las.x), dtype=dtype)


    def fast_patch_mask(self, patch: Patch) -> np.ndarray:
        polygon = patch.shapely_polygon
        min_x, min_y, max_x, max_y = polygon.bounds

        bbox_mask = (self.coords[:, 0] >= min_x) & (self.coords[:, 0] <= max_x) & (self.coords[:, 1] >= min_y) & (self.coords[:, 1] <= max_y)

        center = patch.metadata["center"]
        direction = patch.metadata["direction"]
        length = patch.metadata["length"]
        width = patch.metadata["width"]

        theta = np.arctan2(direction[1], direction[0])

        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])        
        coords = self.coords[bbox_mask].copy()
        coords_shifted = coords - center

        coords_local = coords_shifted @ rotation_matrix

        half_len = length / 2
        half_width = width / 2

        inside_mask = (np.abs(coords_local[:, 0]) <= half_len) & (np.abs(coords_local[:, 1]) <= half_width)
        # Return the indices in the full array
        full_indices = np.where(bbox_mask)[0][inside_mask]
 
        return full_indices



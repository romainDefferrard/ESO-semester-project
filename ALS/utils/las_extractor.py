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

from tqdm import tqdm

class LasExtractor:
    def __init__(self, config: dict, input_file: str, patches: List[Patch]):
        self.extraction_mode = config["EXTRACTION_MODE"]
        self.input_file = input_file
        self.patches = patches  # list of patches instances

        self.las = None  # To store the LAS data (useful for the header)
        self.coords = None  # and the coordinates x,y

        # useful in ascii format
        self.z = None
        self.gps_times = None  # GPS time
        self.intensities = None  # Intensity or Classification

        self.coords_mask = None

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
        """LAS/LAZ reader depending on the format"""
        if not os.path.exists(self.input_file):
            logging.error(f"File not found: {self.input_file}")
            return False
        with laspy.open(self.input_file) as fh:
            self.las = fh.read()  # Read the file fully
            self.coords = np.vstack((self.las.x, self.las.y)).transpose()  # Extract XYZ

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
            self.encode_patches_dynamic(patches, flight_id, output_dir)
        else:
            raise ValueError(f"Unknown extraction mode: {self.extraction_mode}")

    def encode_patches_dynamic(self, patches: List[Patch], flight_id: str, output_dir: str) -> None:
        
        output_file = f"{output_dir}/output_flight_{flight_id}.laz"
        num_points = len(self.coords)

        new_las = self.las
        for dim in self.las.point_format.dimension_names:
            setattr(new_las, dim, getattr(self.las, dim))

        # Init fields
        new_las.add_extra_dim(ExtraBytesParams(name="num_patches", type=np.uint8))
        new_las.num_patches = np.zeros(num_points, dtype=np.uint8)

        new_las.add_extra_dim(ExtraBytesParams(name="patch_ids_1", type=np.uint8))
        new_las.patch_ids_1 = np.zeros(num_points, dtype=np.uint8)

        created_fields = {"patch_ids_1"}
        field_data = {"patch_ids_1": new_las["patch_ids_1"]}

        t0 = time.time()
        for i, patch in enumerate(tqdm(patches, desc=f"Masking flight {flight_id}", unit="patch")):
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
        t1 = time.time()
        logging.info(f"Total time for masking & writing: {t1 - t0:.2f}s")

        # Writing
        t2 = time.time()
        new_las.write(output_file)
        t3 = time.time()
        logging.info(f"Time to write LAS file: {t3 - t2:.2f}s")



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



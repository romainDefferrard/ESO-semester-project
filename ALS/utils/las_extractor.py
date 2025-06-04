"""
Filename: las_extractor.py
Author: Romain Defferrard
Date: 04-06-2025

Description:
    This module defines the LasExtractor class used to extract subsets of point clouds
    from LAS/LAZ or ASCII (.TXYZS) files based on rectangular patch regions. Extraction
    can be done using a fast geometric filter (Transformation in patch frame) on each patch.

    Supported extraction modes:
        - 'binary'      : (RECOMMANDED) saves patch membership in a separate binary file. 
        - 'independent' : saves each patch as a separate file.
        - 'Extra_Bytes' : adds patch membership info as extra dimensions in-place.

    Supported input formats: .las, .laz, .TXYZS, .txt

    This class uses a TimerLogger utility to benchmark steps in the pipeline.
"""


import laspy
import numpy as np
import pandas as pd
import os
import copy
import logging
from laspy import ExtraBytesParams
from typing import List
import copy

from .patch_model import Patch
from .timer_logger import TimerLogger


class LasExtractor:
    def __init__(self, config: dict, input_file: str, patches: List[Patch]):
        """
        Initializes the LasExtractor class.

        Inputs:
            config (dict): Configuration dictionary with keys like EXTRACTION_MODE.
            input_file (str): Path to input LAS/LAZ or ASCII (.TXYZS) file.
            patches (List[Patch]): List of Patch objects to process.

        Output:
            None
        """
        self.extraction_mode = config["EXTRACTION_MODE"]
        self.input_file = input_file
        self.patches = patches  # list of patches instances
        
        # Loaded point cloud content
        self.las = None  
        self.coords = None  #2D XY for masking

        # For ASCII format
        self.z = None
        self.gps_times = None  # GPS time
        self.intensities = None  # Intensity or Classification

        self.coords_mask = None
        
        # Timer utility to measure durations
        self.timer = TimerLogger()


        self.file_format = self.detect_file_format()
        

    def detect_file_format(self) -> str:
        """
        Detects the input file format.

        Output:
            str: 'laz' or 'TXYZS'
        """
        if self.input_file.endswith(".laz") or self.input_file.endswith(".las"):
            return "laz"
        elif self.input_file.endswith(".TXYZS") or self.input_file.endswith(".txt"):
            return "TXYZS"
        else:
            raise ValueError(f"Unsupported file format: {self.input_file}. Supported: .laz, .las, .TXYZS")

    def read_point_cloud(self) -> bool:
        """
        Reads the input point cloud file based on its format.

        Output:
            bool: True if successful, False if file not found.
        """
        if self.file_format == "laz":
            return self.las_read()
        elif self.file_format == "TXYZS":
            return self.ascii_read()

    def las_read(self) -> bool:
        """
        Reads LAS/LAZ file.

        Output:
            bool: True if file exists and read successfully, False otherwise.
        """
        if not os.path.exists(self.input_file):
            logging.error(f"File not found: {self.input_file}")
            return False

        with laspy.open(self.input_file) as fh:
            self.las = fh.read()
            self.coords = np.vstack((self.las.x, self.las.y)).T

        return True


    def ascii_read(self) -> bool:
        """
        Reads ASCII .TXYZS or .txt file.

        Output:
            bool: True if successful, False if file not found or invalid.
        """
        if not os.path.exists(self.input_file):
            logging.error(f"File not found: {self.input_file}")
            return False

        df = pd.read_csv(self.input_file, delimiter="\t", header=None, dtype=float)

        if df.shape[1] < 7:
            raise ValueError("File does not contain enough columns.")

        self.gps_times = df.iloc[:, 0].values  # Time
        self.coords = df.iloc[:, 1:3].values  # X, Y
        self.z = df.iloc[:, 3].values  # Z
        self.intensities = df.iloc[:, 4:7].values  # Intensity values (3 columns)

        return True


    def write_las(self, output_file: str) -> None:
        """
        Writes extracted LAS points to a new file ("independent" extraction case).

        Inputs:
            output_file (str): Path where the new LAS file should be saved.

        Output:
            None
        """
        self.copy_header()
        new_las = laspy.LasData(self.header)
        for dimension in self.las.point_format.dimension_names:
            data = getattr(self.las, dimension)
            setattr(new_las, dimension, data[self.coords_mask])

        new_las.write(output_file)

    def write_ascii(self, output_file: str) -> None:
        """
        Writes extracted points to ASCII .TXYZS format ("independent" extraction case).

        Inputs:
            output_file (str): Path where the .txt file should be saved.

        Output:
            None
        """
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
        """
        Copies the LAS file header and stores the point count.

        Output:
            None
        """
        self.header = copy.deepcopy(self.las.header)
        self.header.point_count = np.sum(self.coords_mask)


    def process_all_patches(self, patches: List[Patch], output_dir: str, flight_id: str, pair_dir: str) -> None:
        """
        Directs the patch extraction process according to the selected extraction mode.

        Inputs:
            patches (List[Patch]): Patches to process.
            output_dir (str): Output directory base path.
            flight_id (str): Current flight identifier.
            pair_dir (str): Output directory for this flight pair.

        Output:
            None
        """
        if self.extraction_mode == "independent":
            self.extract_independant(patches, flight_id, pair_dir)
        elif self.extraction_mode == "Extra_Bytes":
            self.encode_patches_extrabytes(patches, output_dir)
        elif self.extraction_mode == "binary": 
            self.encode_patches_binary(patches, flight_id, output_dir)
        else:
            raise ValueError(f"Unknown extraction mode: {self.extraction_mode}")
        
    def extract_independant(self, patches: List[Patch], flight_id: str, pair_dir: str):
        """
        Extracts and saves each patch to an individual file using geometric filtering.

        Inputs:
            patches (List[Patch]):  List of patch objects to extract.
            flight_id (str):        Identifier of the current flight.
            pair_dir (str):         Output directory path to store patch files.

        Output:
            None
        """
        os.makedirs(pair_dir, exist_ok=True)

        for patch in patches:
            output_file = os.path.join(pair_dir, f"patch_{patch.id}_flight_{flight_id}.{self.file_format}")
            
            selected_indices = self.fast_geometric_mask(patch)
            if len(selected_indices) == 0:
                logging.warning(f"No filtered points in patch {patch.id}, skipping save.")
                continue

            self.coords_mask = np.zeros(len(self.coords), dtype=bool)
            self.coords_mask[selected_indices] = True

            if self.file_format == "TXYZS":
                self.write_ascii(output_file)
            else:
                self.write_las(output_file)    
    
    def encode_patches_extrabytes(self, patches: List[Patch], output_dir: str) -> None:
        """
        Encodes patch membership directly into the LAS file using ExtraBytes dimensions.

        For each patch, this method identifies the points inside it using a fast geometric filter,
        then annotates those points with the patch ID in dedicated extra byte fields (e.g., 'patch_ids_1').
        If a point belongs to multiple patches, additional fields like 'patch_ids_2', 'patch_ids_3', etc. are added.

        A separate counter field 'num_patches' keeps track of how many patches each point belongs to.
        This information is written back into the original LAS file, replacing it in-place.

        Inputs:
            patches (List[Patch]): Patches to encode.
            flight_id (str): Flight identifier.
            output_dir (str): Output directory where the temporary LAS file is written before replacement.

        Output:
            None
        """
        filename = os.path.basename(self.input_file)
        output_laz = os.path.join(output_dir, filename)
        
        new_las = self.las

        self.safe_add_or_reset("num_patches", np.uint8, new_las)
        self.safe_add_or_reset("patch_ids_1", np.uint8, new_las)

        created_fields = {"patch_ids_1"}
        field_data = {"num_patches": new_las["num_patches"],
                    "patch_ids_1": new_las["patch_ids_1"]}

        patch_masks = []
        for patch in patches:
            selected_indices = self.fast_geometric_mask(patch)
            if len(selected_indices) > 0:
                patch_masks.append((patch, selected_indices))

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

        new_las.write(output_laz)
        os.replace(output_laz, self.input_file)  
 
    def encode_patches_binary(self, patches: List[Patch], flight_id: str, output_dir: str) -> None:
        """
        Saves patch membership in a compact binary matrix.

        Each point is tagged with how many patches it belongs to, and to which ones.
        The result is saved to a binary file (.patchbin) with dynamic columns.

        Inputs:
            patches (List[Patch]): Patches to encode.
            flight_id (str): Flight identifier.
            output_dir (str): Destination directory.

        Output:
            None
        """
        output_file = f"{output_dir}/flight_{flight_id}.patchbin"
        num_points = len(self.coords)

        patch_array = np.zeros((num_points, 2), dtype=np.uint8)
        
        patch_masks = []
        for patch in patches:
            selected_indices = self.fast_geometric_mask(patch)
            if len(selected_indices) > 0:
                patch_masks.append((patch, selected_indices))

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

        num_columns = np.uint8(patch_array.shape[1])
        with open(output_file, "wb") as f:
            f.write(np.array([num_columns], dtype=np.uint8).tobytes())  # <- Use this
            f.write(patch_array.tobytes())
            
    

    def safe_add_or_reset(self, name, dtype, las) -> None:
        """
        Adds a new extra dimension to the LAS file or resets it to zero if it already exists.

        Inputs:
            name (str): Name of the extra byte field.
            dtype (type): Data type for the field (e.g., np.uint8).
            las (laspy.LasData): LAS file object.

        Output:
            None
        """
        if name in las.point_format.extra_dimension_names:
            las[name][:] = 0
        else:
            las.add_extra_dim(ExtraBytesParams(name=name, type=dtype))
            las[name] = np.zeros(len(las.x), dtype=dtype)


    def fast_geometric_mask(self, patch: Patch) -> np.ndarray:
        """
        Identifies all point cloud indices that fall within a rotated rectangular patch area.

        This function performs a fast spatial filtering by transforming all candidate points
        (those within the bounding box of the patch in E-N coordinates) into the local coordinate frame 
        of the patch. The local frame is defined such that:
            - The patch center becomes the origin (0, 0).
            - The y-axis aligns with the patch's length direction (using the patch's "direction" vector).
            - The x-axis aligns with the patch's width direction.

        After transformation, the function checks whether each point lies within the rectangular bounds
        defined by the patch length and width.

        Inputs:
            patch (Patch): 
                - shapely_polygon: the bounding polygon of the patch.
                - metadata["center"]: (x, y) coordinates of patch center.
                - metadata["direction"]: unit vector along patch's main axis.
                - metadata["length"]: patch length (along main axis).
                - metadata["width"]: patch width (perpendicular to main axis).

        Output:
            np.ndarray:
                Array of indices (into self.coords) for points that lie within the patch.
        """
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



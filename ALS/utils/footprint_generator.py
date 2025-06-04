"""
Filename: footprint_generator.py
Author: Romain Defferrard
Date: 04-06-2025

Description:
    This module defines the Footprint class, which computes observed raster zones for a set of flight trajectories
    using LiDAR scanning geometry. It generates boolean masks indicating which parts of the raster are visible
    from each flight, based on scan direction, field of view, tilt, and altitude. It finally identifies zones observed 
    by multiple flights (superpositions).
    
    The main outputs are:
        - self.observed_masks: List[np.ndarray] of boolean masks (one per flight).
        - self.superpos_masks: List[np.ndarray] of boolean masks (one per flight pair).
        - self.superpos_flight_pairs: List[Tuple[str, str]] flight ID pairs used in superposition analysis.
"""
import numpy as np
from itertools import combinations, pairwise
from multiprocessing import Pool
from typing import Dict, Tuple
from tqdm import tqdm
from shapely.vectorized import contains
from shapely.geometry import Polygon

def get_footprint_wrapper(args):
    return Footprint.get_footprint(*args)

def build_polygon_from_masks(left_mask, right_mask, x_mesh, y_mesh):
    def extract_coords(mask):
        y_idx, x_idx = np.where(mask)
        x = x_mesh[y_idx, x_idx]
        y = y_mesh[y_idx, x_idx]
        return np.column_stack((x, y))

    left_coords = extract_coords(left_mask)
    right_coords = extract_coords(right_mask)

    # Simple sorting along x or y (e.g. x)
    left_sorted = left_coords[np.argsort(left_coords[:, 0])]
    right_sorted = right_coords[np.argsort(right_coords[:, 0])]

    # Combine into closed loop
    polygon_coords = np.concatenate([left_sorted, right_sorted[::-1]])
    polygon = Polygon(polygon_coords)
    return polygon

class Footprint:
    def __init__(self, raster: np.ndarray, raster_mesh: Tuple[np.ndarray, np.ndarray], flights: Dict[str, dict], config: dict) -> None:
        """
        Initializes the Footprint object and computes visibility masks for all flights.

        Inputs:
            raster (np.ndarray): 2D elevation array (DTM).
            raster_mesh (tuple of np.ndarray): x and y coordinate grids [m].
            flights (dict[str, dict]): Dictionary of flight trajectory data.
            config (dict): Configuration dictionary.
                - PAIR_MODE ("successive" or "all")
                - LIDAR_SCAN_MODE ("left", "right", or "across")
                - LIDAR_TILT_ANGLE (float): LIDAR tilt in degrees
                - LIDAR_FOV (float): Field of view in degrees
                - FLIGHT_DOWNSAMPLING (int): Sampling interval for processing
                - POSITION_BUFFER (float): Buffer in meters to extend front/back of scans

        Outputs:
            - self.observed_masks
            - self.superpos_masks
            - self.superpos_flight_pairs
        """
        self.raster_map = raster
        self.flights = flights
        self.x_mesh, self.y_mesh = raster_mesh

        # Retrieve configuration parameters
        self.mode = config["PAIR_MODE"]
        self.lidar_scan_mode = config["LIDAR_SCAN_MODE"]
        self.lidar_tilt_angle = config["LIDAR_TILT_ANGLE"]
        self.lidar_fov = config["LIDAR_FOV"]
        self.sampling_interval = config["FLIGHT_DOWNSAMPLING"]

        # Masks
        self.superpos_masks = []
        self.observed_masks = []
        self.superpos_flight_pairs = []

        self.get_superpos_zones()

    def get_superpos_zones(self) -> None:
        """
        Computes all footprint masks and overlaps between flights.

        Input:
            None (uses internal attributes).

        Output:
            Updates:
                - self.observed_masks (list of boolean np.ndarray): visibility mask per flight
                - self.superpos_masks (list of boolean np.ndarray): overlap mask per flight pair
                - self.superpos_flight_pairs (list of tuple): corresponding flight ID pairs
        """
        flight_ids = []
        flight_key_order = list(self.flights.keys())

        tasks = [
            (
                flight_key,
                self.flights[flight_key],
                self.raster_map,
                self.x_mesh,
                self.y_mesh,
                self.lidar_scan_mode,
                self.lidar_tilt_angle,
                self.lidar_fov,
                self.sampling_interval,
            )
            for flight_key in flight_key_order
        ]

        with Pool() as pool:
            results = list(tqdm(pool.imap_unordered(get_footprint_wrapper, tasks), total=len(tasks), desc="Génération des footprints"))

        result_dict = dict(results)
        for flight_key in flight_key_order:
            self.observed_masks.append(result_dict[flight_key])
            flight_id = flight_key.split("_")[-1]
            flight_ids.append(flight_id)

        flight_id_to_index = {flight_id: idx for idx, flight_id in enumerate(flight_ids)}

        if self.mode == "successive":
            flight_pairs = pairwise(flight_ids)
        elif self.mode == "all":
            flight_pairs = combinations(flight_ids, 2)
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose 'successive' or 'all'.")

        for flight_id_1, flight_id_2 in flight_pairs:
            idx_1 = flight_id_to_index[flight_id_1]
            idx_2 = flight_id_to_index[flight_id_2]
            combined_mask = self.observed_masks[idx_1] & self.observed_masks[idx_2]
            self.superpos_masks.append(combined_mask)
            self.superpos_flight_pairs.append((flight_id_1, flight_id_2))

    @staticmethod
    def get_footprint(
        flight_key: str,
        flight_data: dict,
        raster_map: np.ndarray,
        x_mesh: np.ndarray,
        y_mesh: np.ndarray,
        lidar_scan_mode: str,
        lidar_tilt_angle: float,
        lidar_fov: float,
        sampling_interval: int,
    ) -> Tuple[str, np.ndarray]:
        """
        This function computes the observed LiDAR footprint for a single flight by identifying, 
        at each sampled position, the farthest visible terrain cell along the left and right edges 
        of the field of view (FOV), based on scanning angles and DTM elevation. These edge points are 
        classified as left or right depending on the scan direction, then combined into a closed polygon 
        representing the footprint boundary. The polygon is finally rasterized into a boolean mask 
        indicating all grid cells inside the observed area.

        Inputs:
            flight_key (str):              Identifier for the flight.
            flight_data (dict):            Flight data with keys 'lon', 'lat', 'alt' as pandas Series.
            raster_map (np.ndarray):       2D array of elevation values.
            x_mesh, y_mesh (np.ndarray):   Grid coordinate arrays matching raster_map.
            lidar_scan_mode (str):         Scan direction ("left", "right", or "across").
            lidar_tilt_angle (float):      LiDAR tilt angle in degrees.
            lidar_fov (float):             Field of view angle in degrees.
            sampling_interval (int):       Interval for downsampling trajectory points.

        Output:
            tuple[str, np.ndarray]: (flight_key, observed_mask), where observed_mask is a boolean np.ndarray.
        """
        half_fov_rad = np.radians(lidar_fov / 2)
        E = flight_data["lon"][::sampling_interval]
        N = flight_data["lat"][::sampling_interval]
        A = flight_data["alt"][::sampling_interval]
 
        def get_scan_angles(trajectory_angle, mode, tilt_rad):
            if mode == "left":
                return trajectory_angle + np.pi / 2 + tilt_rad, trajectory_angle - np.pi / 2 + tilt_rad
            elif mode == "right":
                return trajectory_angle + np.pi / 2 - tilt_rad, trajectory_angle - np.pi / 2 - tilt_rad
            elif mode == "across":
                return trajectory_angle + np.pi / 2, trajectory_angle - np.pi / 2

        def angle_diff(a, b):
            return (a - b + np.pi) % (2 * np.pi) - np.pi

        left_mask_total = np.zeros_like(raster_map, dtype=bool)
        right_mask_total = np.zeros_like(raster_map, dtype=bool)
        max_distance_mask = np.zeros_like(raster_map, dtype=bool)

        for e, n, alt in zip(E, N, A):
            trajectory_angle = np.arctan2(N.iloc[-1] - N.iloc[0], E.iloc[-1] - E.iloc[0])
            scanning_angle_1, scanning_angle_2 = get_scan_angles(
                trajectory_angle, lidar_scan_mode, np.radians(lidar_tilt_angle)
            )

            angle_to_grid = np.arctan2(y_mesh - n, x_mesh - e)
            horizontal_distances = np.sqrt((x_mesh - e) ** 2 + (y_mesh - n) ** 2)
            vertical_distances = alt - raster_map
            line_of_sight_angles = np.arctan2(horizontal_distances, vertical_distances)
            fov_mask = np.abs(line_of_sight_angles) <= half_fov_rad

            tolerance = np.deg2rad(5)

            for scan_angle in [scanning_angle_1, scanning_angle_2]:
                direction_mask = np.abs(angle_diff(angle_to_grid, scan_angle)) <= tolerance
                direction_visible_mask = direction_mask & fov_mask

                if np.any(direction_visible_mask):
                    # Extract indices of valid points
                    idx = np.where(direction_visible_mask)
                    dists = horizontal_distances[idx]
                    max_idx = np.argmax(dists)

                    max_i = idx[0][max_idx]
                    max_j = idx[1][max_idx]

                    max_distance_mask[max_i, max_j] = True

                    # Classify into left/right
                    if scan_angle == scanning_angle_2:
                        left_mask_total[max_i, max_j] = True
                    else:
                        right_mask_total[max_i, max_j] = True

        # --- polygon & mask ---
        polygon = build_polygon_from_masks(left_mask_total, right_mask_total, x_mesh, y_mesh)
        final_mask = contains(polygon, x_mesh, y_mesh)

        return flight_key, final_mask

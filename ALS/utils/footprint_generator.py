"""
Filename: footprint_generator.py
Author: Romain Defferrard
Date: 08-05-2025

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
from shapely.geometry import Polygon
from matplotlib.path import Path

def get_footprint_wrapper(args):
    return Footprint.get_footprint_fast(*args)

class Footprint:
    def __init__(self, raster: np.ndarray, raster_mesh: Tuple[np.ndarray, np.ndarray], flights: Dict[str, dict], config: dict) -> None:
        self.raster_map = raster
        self.flights = flights
        self.x_mesh, self.y_mesh = raster_mesh

        self.mode = config["PAIR_MODE"]
        self.lidar_scan_mode = config["LIDAR_SCAN_MODE"]
        self.lidar_tilt_angle = config["LIDAR_TILT_ANGLE"]
        self.lidar_fov = config["LIDAR_FOV"]
        self.sampling_interval = config["FLIGHT_DOWNSAMPLING"]
        self.buffer_dist = config["POSITION_BUFFER"]

        self.superpos_masks = []
        self.observed_masks = []
        self.superpos_flight_pairs = []

        self.get_superpos_zones()

    def get_superpos_zones(self) -> None:
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
                self.buffer_dist,
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
    def get_footprint_fast(
        flight_key: str,
        flight_data: dict,
        raster_map: np.ndarray,
        x_mesh: np.ndarray,
        y_mesh: np.ndarray,
        lidar_scan_mode: str,
        lidar_tilt_angle: float,
        lidar_fov: float,
        sampling_interval: int,
        buffer_dist: float,
    ) -> Tuple[str, np.ndarray]:

        E = flight_data["lon"][::sampling_interval]
        N = flight_data["lat"][::sampling_interval]
        A = flight_data["alt"][::sampling_interval]

        all_polygons = []

        for i in range(len(E)):
            e, n, alt = E.iloc[i], N.iloc[i], A.iloc[i]

            if i < len(E) - 1:
                next_e, next_n = E.iloc[i+1], N.iloc[i+1]
            else:
                next_e, next_n = E.iloc[i-1], N.iloc[i-1]

            trajectory_angle = np.arctan2(next_n - n, next_e - e)

            half_fov_rad = np.radians(lidar_fov / 2)
            tilt_rad = np.radians(lidar_tilt_angle)

            if lidar_scan_mode == "left":
                angles = [trajectory_angle + np.pi/2 + tilt_rad - half_fov_rad,
                          trajectory_angle + np.pi/2 + tilt_rad + half_fov_rad]
            elif lidar_scan_mode == "right":
                angles = [trajectory_angle - np.pi/2 - tilt_rad - half_fov_rad,
                          trajectory_angle - np.pi/2 - tilt_rad + half_fov_rad]
            elif lidar_scan_mode == "across":
                angles = [trajectory_angle + np.pi/2 - half_fov_rad,
                          trajectory_angle + np.pi/2 + half_fov_rad]

            max_range = 500
            points = [(e, n)]
            for angle in angles:
                x = e + max_range * np.cos(angle)
                y = n + max_range * np.sin(angle)
                points.append((x, y))

            poly = Polygon(points)
            all_polygons.append(poly)

        union_poly = all_polygons[0]
        for poly in all_polygons[1:]:
            union_poly = union_poly.union(poly)

        points = np.vstack((x_mesh.ravel(), y_mesh.ravel())).T
        mask = Path(np.array(union_poly.exterior.coords)).contains_points(points)
        mask = mask.reshape(x_mesh.shape)

        return flight_key, mask

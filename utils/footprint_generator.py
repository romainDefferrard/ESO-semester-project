import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, pairwise
import numpy as np
from multiprocessing import Pool
import time
import logging
from typing import Dict, Tuple, List, Optional


class Footprint:
    def __init__(self, raster: np.ndarray, raster_mesh: Tuple[np.ndarray, np.ndarray], flights: Dict[str, dict], config: dict) -> None:
        self.raster_map = raster
        self.flights = flights
        self.x_mesh, self.y_mesh = raster_mesh

        # Retrieve configuration parameters
        self.mode = config["PAIR_MODE"]
        self.lidar_scan_mode = config["LIDAR_SCAN_MODE"]  # 'left', 'right', 'across'
        self.lidar_tilt_angle = config["LIDAR_TILT_ANGLE"]  # [deg] tilt angle from across track 0deg tilt
        self.lidar_fov = config["LIDAR_FOV"]
        self.sampling_interval = config["FLIGHT_DOWNSAMPLING"]
        self.buffer_dist = config["POSITION_BUFFER"]

        # Masks
        self.superpos_masks = []
        self.observed_masks = []  # not really useful.. maybe for visualization ??
        self.superpos_flight_pairs = []  # to store pairs of flight which have overlaps

        self.get_superpos_zones()

    def create_tasks(self) -> List[Tuple[str, dict]]:
        return [(flight_key, flight_data) for flight_key, flight_data in self.flights.items()]

    def get_superpos_zones(self) -> None:
        flight_ids = []  
        tasks = self.create_tasks()

        with Pool() as pool:
            results = pool.starmap(self.get_footprint, tasks)  
            
        results.sort(key=lambda x: x[0])  # Sort by flight_key so we append masks in right orders

        for flight_key, observed_mask in results:
            self.observed_masks.append(observed_mask)

            flight_id = flight_key.split("_")[-1]
            flight_ids.append(flight_id)  # Append flight ID to list

        flight_id_to_index = {flight_id: idx for idx, flight_id in enumerate(flight_ids)}

        # Define overlap mode
        if self.mode == "successive":
            flight_pairs = pairwise(flight_ids)  # Only successive flight pairs
        elif self.mode == "all":
            flight_pairs = combinations(flight_ids, 2)  # All possible combinations
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose 'successive' or 'all'.")

        # Process flight pairs based on selected mode
        for flight_id_1, flight_id_2 in flight_pairs:
            idx_1 = flight_id_to_index[flight_id_1]
            idx_2 = flight_id_to_index[flight_id_2]
            combined_mask = self.observed_masks[idx_1] & self.observed_masks[idx_2]
            self.superpos_masks.append(combined_mask)
            self.superpos_flight_pairs.append((flight_id_1, flight_id_2))

    def flight_coordinates(self, flight_data: dict) -> None:
        step = self.sampling_interval
        self.flight_E = flight_data["lon"][::step]
        self.flight_N = flight_data["lat"][::step]
        self.flight_alt = flight_data["alt"][::step]

    def get_footprint(self, flight_key: str, flight_data: dict) -> Tuple[str, np.ndarray]:
        """
        Computes the footprint per flight, considering LiDAR scanning mode (across-track, left, right)
        and tilt angle. Uses directional angle checks + FOV filtering.
        """
        half_fov_rad = np.radians(self.lidar_fov / 2)

        self.flight_coordinates(flight_data)

        observed_mask = np.zeros_like(self.raster_map, dtype=bool)

        # Iterate over each flight position (e, n, alt)
        for i, (e, n, alt) in enumerate(zip(self.flight_E, self.flight_N, self.flight_alt)):

            # Compute the trajectory angle of the aircraft
            trajectory_angle = np.arctan2(self.flight_N.iloc[-1] - self.flight_N.iloc[0], self.flight_E.iloc[-1] - self.flight_E.iloc[0])

            # **Compute Scanning Angles (Two Directions)**
            if self.lidar_scan_mode == "left":
                scanning_angle_1 = trajectory_angle + np.pi / 2 + np.radians(self.lidar_tilt_angle)  # Forward-right
                scanning_angle_2 = trajectory_angle - np.pi / 2 + np.radians(self.lidar_tilt_angle)  # Backward-left

            elif self.lidar_scan_mode == "right":
                scanning_angle_1 = trajectory_angle + np.pi / 2 - np.radians(self.lidar_tilt_angle)  # Forward-left
                scanning_angle_2 = trajectory_angle - np.pi / 2 - np.radians(self.lidar_tilt_angle)  # Backward-right

            elif self.lidar_scan_mode == "across":
                scanning_angle_1 = trajectory_angle + np.pi / 2  # Left
                scanning_angle_2 = trajectory_angle - np.pi / 2  # Right

            if i == 0:  # add only buffer forward
                e_avant = e + self.buffer_dist * np.cos(trajectory_angle)
                n_avant = n + self.buffer_dist * np.sin(trajectory_angle)
                e_arriere = e
                n_arriere = n

            elif i == len(self.flight_E) - 1:  # add only buffer backward
                e_avant = e
                n_avant = n
                e_arriere = e - self.buffer_dist * np.cos(trajectory_angle)
                n_arriere = n - self.buffer_dist * np.sin(trajectory_angle)

            else:  # add buffer both forward and backward
                e_avant = e + self.buffer_dist * np.cos(trajectory_angle)
                n_avant = n + self.buffer_dist * np.sin(trajectory_angle)
                e_arriere = e - self.buffer_dist * np.cos(trajectory_angle)
                n_arriere = n - self.buffer_dist * np.sin(trajectory_angle)

            # **Recalcul des angles des tuiles par rapport aux positions avancée et reculée**
            angle_to_grid_forward = np.arctan2(self.y_mesh - n_avant, self.x_mesh - e_avant)
            angle_to_grid_backward = np.arctan2(self.y_mesh - n_arriere, self.x_mesh - e_arriere)

            def is_between(angle, min_angle, max_angle):
                return ((angle - min_angle) % (2 * np.pi)) < ((max_angle - min_angle) % (2 * np.pi))

            valid_scan_mask = is_between(angle_to_grid_forward, scanning_angle_1, scanning_angle_2) & is_between(
                angle_to_grid_backward, scanning_angle_2, scanning_angle_1
            )

            # **Compute FOV filtering**
            horizontal_distances = np.sqrt((self.x_mesh - e) ** 2 + (self.y_mesh - n) ** 2)
            vertical_distances = alt - self.raster_map
            line_of_sight_angles = np.arctan2(horizontal_distances, vertical_distances)

            # Ensure points are within the FOV
            fov_mask = np.abs(line_of_sight_angles) <= half_fov_rad

            # **Final mask: points must be within FOV & align with scanning direction**
            observed_mask |= valid_scan_mask & fov_mask

        return flight_key, observed_mask


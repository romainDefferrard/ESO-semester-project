import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, pairwise
from multiprocessing import Pool
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm


def get_footprint_wrapper(args):
    return Footprint.get_footprint_static(*args)


class Footprint:
    def __init__(self, raster: np.ndarray, raster_mesh: Tuple[np.ndarray, np.ndarray], flights: Dict[str, dict], config: dict) -> None:
        self.raster_map = raster
        self.flights = flights
        self.x_mesh, self.y_mesh = raster_mesh

        # Retrieve configuration parameters
        self.mode = config["PAIR_MODE"]
        self.lidar_scan_mode = config["LIDAR_SCAN_MODE"]
        self.lidar_tilt_angle = config["LIDAR_TILT_ANGLE"]
        self.lidar_fov = config["LIDAR_FOV"]
        self.sampling_interval = config["FLIGHT_DOWNSAMPLING"]
        self.buffer_dist = config["POSITION_BUFFER"]

        # Masks
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
    def get_footprint_static(
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

        half_fov_rad = np.radians(lidar_fov / 2)
        E = flight_data["lon"][::sampling_interval]
        N = flight_data["lat"][::sampling_interval]
        A = flight_data["alt"][::sampling_interval]

        observed_mask = np.zeros_like(raster_map, dtype=bool)

        for i, (e, n, alt) in enumerate(zip(E, N, A)):
            trajectory_angle = np.arctan2(N.iloc[-1] - N.iloc[0], E.iloc[-1] - E.iloc[0])

            if lidar_scan_mode == "left":
                scanning_angle_1 = trajectory_angle + np.pi / 2 + np.radians(lidar_tilt_angle)
                scanning_angle_2 = trajectory_angle - np.pi / 2 + np.radians(lidar_tilt_angle)
            elif lidar_scan_mode == "right":
                scanning_angle_1 = trajectory_angle + np.pi / 2 - np.radians(lidar_tilt_angle)
                scanning_angle_2 = trajectory_angle - np.pi / 2 - np.radians(lidar_tilt_angle)
            elif lidar_scan_mode == "across":
                scanning_angle_1 = trajectory_angle + np.pi / 2
                scanning_angle_2 = trajectory_angle - np.pi / 2

            if i == 0:
                e_avant = e + buffer_dist * np.cos(trajectory_angle)
                n_avant = n + buffer_dist * np.sin(trajectory_angle)
                e_arriere = e
                n_arriere = n
            elif i == len(E) - 1:
                e_avant = e
                n_avant = n
                e_arriere = e - buffer_dist * np.cos(trajectory_angle)
                n_arriere = n - buffer_dist * np.sin(trajectory_angle)
            else:
                e_avant = e + buffer_dist * np.cos(trajectory_angle)
                n_avant = n + buffer_dist * np.sin(trajectory_angle)
                e_arriere = e - buffer_dist * np.cos(trajectory_angle)
                n_arriere = n - buffer_dist * np.sin(trajectory_angle)

            angle_to_grid_forward = np.arctan2(y_mesh - n_avant, x_mesh - e_avant)
            angle_to_grid_backward = np.arctan2(y_mesh - n_arriere, x_mesh - e_arriere)

            def is_between(angle, min_angle, max_angle):
                return ((angle - min_angle) % (2 * np.pi)) < ((max_angle - min_angle) % (2 * np.pi))

            valid_scan_mask = is_between(angle_to_grid_forward, scanning_angle_1, scanning_angle_2) & \
                              is_between(angle_to_grid_backward, scanning_angle_2, scanning_angle_1)

            horizontal_distances = np.sqrt((x_mesh - e) ** 2 + (y_mesh - n) ** 2)
            vertical_distances = alt - raster_map
            line_of_sight_angles = np.arctan2(horizontal_distances, vertical_distances)
            fov_mask = np.abs(line_of_sight_angles) <= half_fov_rad

            observed_mask |= valid_scan_mask & fov_mask

        return flight_key, observed_mask

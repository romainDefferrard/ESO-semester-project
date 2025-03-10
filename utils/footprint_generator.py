import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, pairwise
import numpy as np
from multiprocessing import Pool
import time 
import logging
from scipy.ndimage import binary_dilation


LIDAR_FOV_DEG = 75
HALF_FOV_RAD = np.radians(LIDAR_FOV_DEG/2)

class Footprint:
    def __init__(self, raster, raster_mesh, flights, mode='all', lidar_scan_mode="across", lidar_tilt_angle=0):
        self.raster_map = raster
        self.flights = flights
        self.x_mesh, self.y_mesh = raster_mesh
        self.mode = mode
        self.lidar_scan_mode = lidar_scan_mode # 'left', 'right', 'across' 
        self.lidar_tilt_angle = lidar_tilt_angle # [deg] tilt angle from across track 0deg tilt

        # Masks 
        self.superpos_masks = []
        self.observed_masks = [] # not really useful.. maybe for visualization ??
        self.superpos_flight_pairs = [] # to store pairs of flight which have overlaps

        self.get_superpos_zones() 
    
    #    self.plot_zones(self.observed_masks[0], self.observed_masks[1], self.superpos_masks[0], flights)
    
    
    def create_tasks(self):
        """Create tasks for parallel execution of `get_footprint()`"""
        return [(flight_key, flight_data) for flight_key, flight_data in self.flights.items()]
        
    def get_superpos_zones(self):
        flight_ids = []  # Ensure it's empty at the start
        
        tasks = self.create_tasks()
        
        with Pool() as pool: 
            results = pool.starmap(self.get_footprint2, tasks) # footprint ou footprint2 (avec scan incliné)

        results.sort(key=lambda x: x[0])  # Sort by flight_key so we append masks in right orders 
 
        for flight_key, observed_mask in results:
            self.observed_masks.append(observed_mask)

            flight_id = flight_key.split("_")[-1]
            flight_ids.append(flight_id)  # Append flight ID to list

        flight_id_to_index = {flight_id: idx for idx, flight_id in enumerate(flight_ids)}

        # Define overlap mode
        match self.mode:
            case "successive":
                flight_pairs = pairwise(flight_ids)  # Only successive flight pairs
            case "all":
                flight_pairs = combinations(flight_ids, 2)  # All possible combinations
            case _:
                raise ValueError(f"Invalid mode: {self.mode}. Choose 'successive' or 'all'.")

        # Process flight pairs based on selected mode
        for flight_id_1, flight_id_2 in flight_pairs:
            idx_1 = flight_id_to_index[flight_id_1]
            idx_2 = flight_id_to_index[flight_id_2]

            # Create a combined mask for this pair of flights
            combined_mask = self.observed_masks[idx_1] & self.observed_masks[idx_2]

            # Store results
            self.superpos_masks.append(combined_mask)
            self.superpos_flight_pairs.append((flight_id_1, flight_id_2))

                  
    def flight_coordinates(self, flight_data):
            self.flight_E = flight_data['lon']
            self.flight_N = flight_data['lat']
            self.flight_alt = flight_data['alt']
             

    def get_footprint2(self, flight_key, flight_data):
        """
        Computes the footprint per flight, considering LiDAR scanning mode (across-track, left, right) 
        and tilt angle. Uses directional angle checks + FOV filtering.
        """
        self.flight_coordinates(flight_data)
        
        observed_mask = np.zeros_like(self.raster_map, dtype=bool)

        # Margin for valid angles
        angle_margin = np.radians(2)  

        # Iterate over each flight position (e, n, alt)
        for e, n, alt in zip(self.flight_E, self.flight_N, self.flight_alt):
            
            # Compute the trajectory angle of the aircraft
            trajectory_angle = np.arctan2(self.flight_N.iloc[-1] - self.flight_N.iloc[0], 
                                        self.flight_E.iloc[-1] - self.flight_E.iloc[0])

            # **Compute Scanning Angles (Two Directions)**
            if self.lidar_scan_mode == "left":
                scanning_angle_1 = trajectory_angle + np.pi / 2 + np.radians(self.lidar_tilt_angle)  # Forward-right
                scanning_angle_2 = trajectory_angle - np.pi / 2 + np.radians(self.lidar_tilt_angle)  # Backward-left

            elif self.lidar_scan_mode == "right":
                scanning_angle_1 = trajectory_angle + np.pi / 2 - np.radians(self.lidar_tilt_angle)  # Forward-left
                scanning_angle_2 = trajectory_angle - np.pi / 2 - np.radians(self.lidar_tilt_angle)  # Backward-right

            elif self.lidar_scan_mode == "across":
                scanning_angle_1 = trajectory_angle + np.pi / 2   # Left
                scanning_angle_2 = trajectory_angle - np.pi / 2   # Right

            # **Compute angle from aircraft to each grid tile**
            angle_to_grid = np.arctan2(self.y_mesh - n, self.x_mesh - e)

            def angle_difference(angle1, angle2):
                """Compute the minimal angle difference considering wrap-around (360° handling)."""
                return np.abs((angle1 - angle2) % (2 * np.pi))

            valid_scan_1 = (angle_difference(angle_to_grid, scanning_angle_1) <= angle_margin)
            valid_scan_2 = (angle_difference(angle_to_grid, scanning_angle_2) <= angle_margin)
            
            valid_scan_mask = valid_scan_1 | valid_scan_2  # Combine both directions

            # **Compute FOV filtering**
            horizontal_distances = np.sqrt((self.x_mesh - e)**2 + (self.y_mesh - n)**2)
            vertical_distances = alt - self.raster_map
            line_of_sight_angles = np.arctan2(horizontal_distances, vertical_distances)

            # Ensure points are within the FOV
            fov_mask = np.abs(line_of_sight_angles) <= HALF_FOV_RAD

            # **Final mask: points must be within FOV & align with scanning direction**
            observed_mask |= (valid_scan_mask & fov_mask)

        #morphological dilation (in case of pixel within the zone not classified)
        observed_mask = binary_dilation(observed_mask, iterations=1)  # Apply a single dilation step
        
        return flight_key, observed_mask


                        
    def get_footprint(self, flight_key, flight_data):
        """
        Footprint per flight 
        """
        self.flight_coordinates(flight_data)

        observed_mask = np.zeros_like(self.raster_map, dtype=bool)
        
        # Correction simple de la FOV effective
        HALF_FOV_CORR = np.cos(np.radians(self.lidar_tilt_angle)) * HALF_FOV_RAD

        # Mask to highlight observed zones based on current flight data
        for e, n, alt in zip(self.flight_E, self.flight_N, self.flight_alt):
            # Horizontal distances for each grid point
            horizontal_distances = np.sqrt((self.x_mesh - e)**2 + (self.y_mesh - n)**2)

            # Vertical distances for each grid point
            vertical_distances = alt - self.raster_map  

            # Angle of the line of sight to each grid point
            line_of_sight_angles = np.arctan2(horizontal_distances, vertical_distances)

            # Update the observed_mask for this flight
            observed_mask |= (np.abs(line_of_sight_angles) <= HALF_FOV_CORR)
            
        # Remove start and end zones (perpendicular sampling lines)    
        cropped_mask = self.crop_footprint(observed_mask)
    #    self.observed_masks.append(cropped_mask) # plus utile avec multiprocessing
        return flight_key, cropped_mask
    
        
    def crop_footprint(self, mask):
        """
        Refined cropping to better match the LiDAR footprint with the actual point cloud coverage.
        Handles trajectory angles properly to avoid incorrect cropping for different flight directions.
        """
        # Compute the direction (heading) of the plane
        flight_dir = np.diff(np.column_stack((self.flight_E, self.flight_N)), axis=0)  
        flight_dir /= np.linalg.norm(flight_dir, axis=1)[:, np.newaxis]  # Normalize to unit vectors

        # Add the last point again to maintain the same length for direction arrays
        flight_dir = np.vstack((flight_dir, flight_dir[-1]))

        # Compute trajectory angles for start and end
        trajectory_angle_start = np.arctan2(flight_dir[0, 1], flight_dir[0, 0])
        trajectory_angle_end = np.arctan2(flight_dir[-1, 1], flight_dir[-1, 0])

        #print(f"Start Trajectory Angle: {trajectory_angle_start:.2f} rad, End: {trajectory_angle_end:.2f} rad")

        # Get flight start and stop positions
        pos_x_start, pos_y_start = self.flight_E.iloc[0], self.flight_N.iloc[0]
        pos_x_stop, pos_y_stop = self.flight_E.iloc[-1], self.flight_N.iloc[-1]

        # Compute the angle of each grid point relative to the start and stop positions
        angle_to_grid_start = np.arctan2(self.y_mesh - pos_y_start, self.x_mesh - pos_x_start)
        angle_to_grid_stop = np.arctan2(self.y_mesh - pos_y_stop, self.x_mesh - pos_x_stop)

        # **Fix: Compute the wrapped angle difference properly**
        def angle_difference(angle1, angle2):
            """Compute the smallest difference between two angles, considering wrap-around at ±π."""
            return np.abs((angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi)

        angle_diff_start = angle_difference(angle_to_grid_start, trajectory_angle_start)
        angle_diff_stop = angle_difference(angle_to_grid_stop, trajectory_angle_end)

        # Clear behind the flight path at the start
        mask &= ~(angle_diff_start > np.pi / 2)

        # Clear ahead of the flight path at the end
        mask &= ~(angle_diff_stop < np.pi / 2)

        # **Refining lateral cropping**
        # Define a lateral distance based on LiDAR FOV and altitude
        lidar_half_fov_rad = np.radians(LIDAR_FOV_DEG / 2)
        max_lateral_distance = np.tan(lidar_half_fov_rad) * np.max(self.flight_alt)

        # Compute perpendicular distances from flight path
        cross_track_dist = np.abs(
            (self.x_mesh - self.flight_E.iloc[0]) * np.sin(trajectory_angle_start) -
            (self.y_mesh - self.flight_N.iloc[0]) * np.cos(trajectory_angle_start)
        )

        # Remove areas that are too far laterally
        mask &= (cross_track_dist <= max_lateral_distance)

        return mask


    
    
    def plot_zones(self, mask_i, mask_j, combined, flights):
    
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the terrain
        ax.pcolormesh(self.x_mesh, self.y_mesh, self.raster_map, cmap='Greens', alpha=0.7)

        # Overlay the observed mask in red
        #ax.pcolormesh(
        #    self.x_mesh, self.y_mesh, np.where(mask_i, self.raster_map, np.nan), cmap='Reds'
        #)
        ax.pcolormesh(
            self.x_mesh, self.y_mesh, np.where(mask_j, self.raster_map, np.nan), cmap='Blues'
        )
        #ax.pcolormesh(
        #    self.x_mesh, self.y_mesh, np.where(combined, self.raster_map, np.nan), cmap='Reds', shading='auto'
        #)

        # Plus ajouter quelque chose qui génère le bon nombre de couleur en fonction du nombre de vols 
        # qu'on a...
        colors = ['blue', 'purple', 'green']
        for i, (flight_key, flight_data) in enumerate(self.flights.items()):

            ax.scatter(flight_data['lon'], flight_data['lat'], color=colors[i % len(colors)], label=f"{flight_key} path", s=5) 

        # Adding labels and title
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_aspect('equal')
        ax.legend()

        plt.show()
     
        """def crop_footprint(self, mask):
 
        #Moyen de faire ça bien plus clean

        # Compute the direction (heading) of the plane
        flight_dir = np.diff(np.column_stack((self.flight_E, self.flight_N)), axis=0)  # (dx, dy) between consecutive points
        flight_dir /= np.linalg.norm(flight_dir, axis=1)[:, np.newaxis]  # Normalize to unit vectors

        # Add the last point again to maintain the same length for direction arrays
        flight_dir = np.vstack((flight_dir, flight_dir[-1]))
        
        dir_x_start, dir_y_start = flight_dir[0]
        pos_x_start, pos_y_start = (self.flight_E.iloc[0], self.flight_N.iloc[0])

        trajectory_angle = np.arctan2(dir_x_start, dir_y_start)
        angle_to_grid = np.arctan2(self.x_mesh - pos_x_start, self.y_mesh - pos_y_start)
        angle_diff = np.abs(angle_to_grid - trajectory_angle)

        mask &= ~(angle_diff > np.pi / 2)  # Clear behind the flight path

        dir_x_stop, dir_y_stop = flight_dir[-1]
        pos_x_stop, pos_y_stop = (self.flight_E.iloc[-1], self.flight_N.iloc[-1])

        trajectory_angle = np.arctan2(dir_x_stop, dir_y_stop)
        angle_to_grid = np.arctan2(self.x_mesh - pos_x_stop, self.y_mesh - pos_y_stop)
        angle_diff = np.abs(angle_to_grid - trajectory_angle)

        mask &= ~(angle_diff < np.pi / 2)  # Clear ahead of end 
        return mask 
        """

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

LIDAR_FOV_DEG = 75
LIDAR_FOV_RAD = np.radians(LIDAR_FOV_DEG/2)

class Footprint:
    def __init__(self, raster, raster_mesh, flights):
        self.raster_map = raster
        self.flights = flights
        self.x_mesh, self.y_mesh = raster_mesh

        # Masks 
        self.superpos_masks = []
        self.observed_masks = [] # not really useful.. maybe for visualization ??
        self.superpos_flight_pairs = [] # to store pairs of flight which have overlaps

        self.get_superpos_zones() 
        
        # Plot superpos zone
        #self.plot_zones()
    
    def get_superpos_zones(self):
        flight_ids = []  # Ensure it's empty at the start

        for flight_key, flight_data in self.flights.items():
            # Get single flight coordinates
            self.flight_coordinates(flight_data)
            # Get single flight footprint
            self.get_footprint()

            flight_id = flight_key.split("_")[-1]
            flight_ids.append(flight_id)  # Append flight ID to list

        flight_id_to_index = {flight_id: idx for idx, flight_id in enumerate(flight_ids)}

        # Get all possible combinations of flights for superposition
        for flight_id_1, flight_id_2 in combinations(flight_ids, 2):
            # Use mapping to get correct indices
            idx_1 = flight_id_to_index[flight_id_1]
            idx_2 = flight_id_to_index[flight_id_2]

            # Create a combined mask for this pair of flights
            combined_mask = self.observed_masks[idx_1] & self.observed_masks[idx_2]

            # Store the combined mask
            self.superpos_masks.append(combined_mask)

            # Store which flights created this mask (keep flight names, not indices)
            self.superpos_flight_pairs.append((flight_id_1, flight_id_2))

                  
    def flight_coordinates(self, flight_data):
            self.flight_E = flight_data['lon']
            self.flight_N = flight_data['lat']
            self.flight_alt = flight_data['alt']
                        
    def get_footprint(self):
        """
        Footprint per flight 
        """
        observed_mask = np.zeros_like(self.raster_map, dtype=bool)
        
        # Mask to highlight observed zones based on current flight data
        for e, n, alt in zip(self.flight_E, self.flight_N, self.flight_alt):
            # Horizontal distances for each grid point
            horizontal_distances = np.sqrt((self.x_mesh - e)**2 + (self.y_mesh - n)**2)

            # Vertical distances for each grid point
            vertical_distances = alt - self.raster_map  

            # Angle of the line of sight to each grid point
            line_of_sight_angles = np.arctan2(horizontal_distances, vertical_distances)

            # Update the observed_mask for this flight
            observed_mask |= (np.abs(line_of_sight_angles) <= LIDAR_FOV_RAD)
            
        # Remove start and end zones (perpendicular sampling lines)    
        cropped_mask = self.crop_footprint(observed_mask)
        self.observed_masks.append(cropped_mask)

            
    def crop_footprint(self, mask):
        """
        Moyen de faire ça bien plus clean
        """
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
    
    def plot_zones(self, mask_i, mask_j, combined, flights):
    
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the terrain
        ax.pcolormesh(self.x_mesh, self.y_mesh, self.raster_map, cmap='Greens', alpha=0.7)

        # Overlay the observed mask in red
        ax.pcolormesh(
            self.x_mesh, self.y_mesh, np.where(mask_i, self.raster_map, np.nan), cmap='Blues', shading='auto'
        )
        ax.pcolormesh(
            self.x_mesh, self.y_mesh, np.where(mask_j, self.raster_map, np.nan), cmap='Purples', shading='auto'
        )
        ax.pcolormesh(
            self.x_mesh, self.y_mesh, np.where(combined, self.raster_map, np.nan), cmap='Reds', shading='auto'
        )

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
     
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from .raster_loader import RasterLoader


LIDAR_FOV_DEG = 75
LIDAR_FOV_RAD = np.radians(LIDAR_FOV_DEG/2)

class Footprint_V2:
    def __init__(self, flights_data, DTM):
        self.flights_data = flights_data

        self.flights = flights_data.flights
        
        # Get flight bound for each flight
        self.flight_bounds = []
        self.compute_flight_bounds()
        
        # Generate raster 
        self.rasters = []
        self.DTM = DTM
        self.generate_rasters()


        # Masks 
        self.superpos_masks = []
        self.observed_masks = [] # not really useful.. maybe for visualization ??
        self.superpos_flight_pairs = [] # to store pairs of flight which have overlaps
        
        self.get_superpos_zones() 
        
        
    def compute_flight_bounds(self):
        for flight_key, flight_data in self.flights.items():
            E_min, E_max = flight_data['lon'].min(), flight_data['lon'].max()
            N_min, N_max = flight_data['lat'].min(), flight_data['lat'].max()
            self.flight_bounds.append([E_min, E_max, N_min, N_max])
                                    
    def generate_rasters(self): # A combiner les deux fonctions plus tard 
        for i in range(len(self.flight_bounds)):
            raster_loader = RasterLoader(self.DTM, epsg=2056, flight_bounds=self.flight_bounds[i])
            self.rasters.append(raster_loader.raster)



    def get_superpos_zones(self):
        for flight_key, flight_data in self.flights.items():
            
            # Get single flight coordinates 
            self.flight_coordinates(flight_data)
            # Get single flight footprint
            self.get_footprint()
            
        # Get all possible combinations of flights for superposition
        flight_number = list(range(1, len(self.observed_masks)+1))
        
        for i, j in combinations(flight_number, 2):
            # Create a combined mask for this pair of flights
            combined_mask = self.observed_masks[i-1] & self.observed_masks[j-1] 
            # Store the combined mask
            self.superpos_masks.append(combined_mask)
            # Store which flights created this mask
            self.superpos_flight_pairs.append((i, j))
            
            # Plot each superposition zone
        #    self.plot_zones(mask_i=self.observed_masks[i-1], mask_j=self.observed_masks[j-1], combined=combined_mask, flights=(i,j))
                        
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
     
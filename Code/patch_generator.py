import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours, approximate_polygon
from shapely.geometry import Polygon, Point, LineString


"""
A partir des zones de superposition on joue avec les polygones 
pour créer les patchs samplés sur la ligne directice de la zone. 
"""

class PatchGenerator():
    def __init__(self, superpos_zones, raster_mesh, raster):
        self.superpos_zones = superpos_zones 
        self.raster_map = raster # Only useful for plotting 
        self.x_mesh, self.y_mesh = raster_mesh
        
        self.tol = 0.05 # tolerance parameter for the contour generation
        self.contour_coords = None
        
        self.get_contour()
        #self.plot_contour()
        
        
    def get_contour(self):
        """
        Faire truc générique pour plusieurs layer dans la variable self.superpos_zones
        """
        contour = find_contours(self.superpos_zones.astype(int))[0]
        coords = approximate_polygon(contour, tolerance=self.tol)
        
        contour_x = coords[:, 1].astype(int)  # Column indices
        contour_y = coords[:, 0].astype(int)  # Row indices
        
        # Convert indices to Swiss coordinates
        self.contour_coords = np.array([self.x_mesh[contour_y, contour_x]+25/2, 
                                        self.y_mesh[contour_y, contour_x]-25/2]).T
        
    def get_centerline(self):
        pass


    def plot_contour(self):
        
        fig, ax = plt.subplots(figsize=(10, 8))

        # Terrain
        ax.pcolormesh(self.x_mesh, self.y_mesh, self.raster_map, cmap='Greens', alpha=0.7)

        # Plot masks
        ax.pcolormesh(self.x_mesh, self.y_mesh, np.where(self.superpos_zones, self.raster_map, np.nan), cmap='Reds', shading='auto')

        # Plot of contour
        ax.plot(self.contour_coords[:,0], self.contour_coords[:,1], '--', color='black', label='Contour')

        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_aspect('equal')
        ax.set_title('normal avec modif ±res/2')
        ax.legend()

        plt.show()
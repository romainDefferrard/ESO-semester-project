import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours, approximate_polygon
from shapely.geometry import Polygon, LineString
from sklearn.decomposition import PCA


"""
A partir des zones de superposition on joue avec les polygones 
pour créer les patchs samplés sur la ligne directice de la zone. 
"""

class PatchGenerator():
    def __init__(self, superpos_zones, raster_mesh, raster, patch_params):
        self.superpos_zones = superpos_zones 
        self.raster_map = raster # Only useful for plotting 
        self.x_mesh, self.y_mesh = raster_mesh
        self.band_length, self.band_width, self.sample_distance = patch_params
        
        self.tol = 0.05 # tolerance parameter for the contour generation
        self.contour = None
        
        # output
        self.patches = []
        
        self.get_contour()
        # Plot contour of superposition zone
        #self.plot_contour()
        centerline = self.get_centerline()
        self.patches_along_centerline(centerline)
        # Plot centerline and patches
        #self.plot_patches(centerline)
        self.patch_to_polygon()
        
        
    def get_contour(self):
        """
        Faire truc générique pour plusieurs layer dans la variable self.superpos_zones
        """
        contour_bulk = find_contours(self.superpos_zones.astype(int))[0]
        coords = approximate_polygon(contour_bulk, tolerance=self.tol)
        
        contour_x = coords[:, 1].astype(int)  # Column indices
        contour_y = coords[:, 0].astype(int)  # Row indices
        
        # Convert indices to Swiss coordinates
        self.contour = np.array([self.x_mesh[contour_y, contour_x]+25/2, 
                                        self.y_mesh[contour_y, contour_x]-25/2]).T
        
    def get_centerline(self):
        mask_coords = np.column_stack(np.where(self.superpos_zones))
        
        # Convert to original coordinate system
        coord_points = np.array([
            self.x_mesh[mask_coords[:, 0], mask_coords[:, 1]],
            self.y_mesh[mask_coords[:, 0], mask_coords[:, 1]]
        ]).T
        
        # Apply PCA to find the principal axis
        pca = PCA(n_components=2)
        pca.fit(coord_points)
        
        # Get the extent of data along the principal component
        projected = pca.transform(coord_points)
        min_proj = projected[:, 0].min()
        max_proj = projected[:, 0].max()
        
        # Generate points along principal axis
        line_points = np.zeros((100, 2))
        line_points[:, 0] = np.linspace(min_proj, max_proj, 100)
        
        # Transform back to original space
        centerline = pca.inverse_transform(line_points)
        
        return centerline
    

    
    def patches_along_centerline(self, centerline):
        # Convert contour to Shapely polygon
        
        contour_polygon = Polygon(self.contour)
        
        # Convert centerline to Shapely LineString
        centerline_line = LineString(centerline)

        if len(centerline) < 2:
            raise ValueError("Centerline must have at least two points")

        direction = centerline[-1] - centerline[0]
        direction = direction / np.linalg.norm(direction)  # Normalize

        # perpendicular direction = cst since line (A VOIR SI ON CHANGE)
        perp_direction = np.array([-direction[1], direction[0]])    
        
        # Find valid starting point
        valid_start_found = False
        start_dist = 0
        
        while not valid_start_found and start_dist < centerline_line.length:
            
            # Get point at current distance along the centerline
            start_point = np.array(centerline_line.interpolate(start_dist).coords[0])
            
            # Create first patch
            patch = self.create_patch(start_point, direction, perp_direction, self.band_length, self.band_width)
            
            patch_poly = Polygon(patch)
            
        #    print(f"Checking patch at start_dist {start_dist}")
            # Check if patch intersects with contour
            if patch_poly.within(contour_polygon):
        #        print(f"Patch at {start_dist} is inside the contour!")
                valid_start_found = True
                self.patches.append(patch)

            else:
                # Move start point forward
                start_dist += 20  # changer incrémentation?
                
            # Message bizarre si j'ai un += 20 pas de message et si autre -> message 
            # Après ça marche très bien même en recevant le message...
        
        # If we couldn't find a valid starting point, return empty list
        if not valid_start_found:
            # ++ add error message 
            return []
        
        # Generate subsequent patches
        current_dist = start_dist + self.sample_distance
        
        # Assume constant direction -> see if we modif the center line to something not linear?
        # Add condition to see it middle patches are too wide, if so -> reduce the width
        while current_dist < centerline_line.length:
            # Get point at current distance along the centerline
            current_point = np.array(centerline_line.interpolate(current_dist).coords[0])
            
            # Create patch
            patch = self.create_patch(current_point, direction, perp_direction, self.band_length, self.band_width)
            self.patches.append(patch)
            
            # Move to next position
            current_dist += self.sample_distance
            
        # check for last patch not to intersect the contour, remove it if it does 
        last_patch = self.patches[-1]
        last_patch_poly = Polygon(last_patch)
        if not last_patch_poly.within(contour_polygon):
            self.patches.pop()
        
        
    
    def create_patch(self, center, direction, perp_direction, length, width):
        # Calculate half-dimensions
        half_length = length / 2
        half_width = width / 2
        
        # Calculate four corners of the rectangle
        corner1 = center + half_length * direction + half_width * perp_direction
        corner2 = center + half_length * direction - half_width * perp_direction
        corner3 = center - half_length * direction - half_width * perp_direction
        corner4 = center - half_length * direction + half_width * perp_direction
        
        # Create patch (closed polygon)
        patch = np.array([corner1, corner2, corner3, corner4, corner1])
        
        return patch
    
    def patch_to_polygon(self):
        self.patches = [Polygon(patch) for patch in self.patches]
        
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
        
    def plot_patches(self, centerline):
        plt.figure(figsize=(10, 8))

        # Plot contour
        plt.plot(self.contour[:, 0], self.contour[:, 1], 'black', label='Contour')

        # Plot centerline
        plt.plot(centerline[:, 0], centerline[:, 1], 'b-', label='Centerline')

        # Plot patches
        for i, patch in enumerate(self.patches):
            plt.plot(patch[:, 0], patch[:, 1], 'g-', alpha=0.7, label=f'Patch {i+1}' if i==0 else "")

        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()
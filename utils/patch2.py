import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours, approximate_polygon
from shapely.geometry import Polygon, LineString, Point
from sklearn.decomposition import PCA
import warnings

"""
A partir des zones de superposition on joue avec les polygones 
pour créer les patchs samplés sur la ligne directice de la zone. 
"""

class PatchGenerator():
    def __init__(self, superpos_zones, raster_mesh, raster, patch_params):
        self.superpos_zones_all = superpos_zones 
        self.raster_map = raster # Only useful for plotting 
        self.x_mesh, self.y_mesh = raster_mesh
        self.band_length, self.band_width, self.sample_distance = patch_params
        
        self.tol = 0.05 # tolerance parameter for the contour generation
        self.contour = None
        
        # Pas trouvé d'autre solution
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely.predicates")
                
        # Output
        self.patches_list = []
        self.centerlines_list = []
        self.contours_list = []
        self.max_patch_len = []
        self.patches_poly_list = []
        
        # Process all superposition zones
        self.process_zones()
        
        # compute big single patch
        
                
    def process_zones(self):
        """Process each superposition zone to generate centerlines and patches."""
        for superpos_zone in self.superpos_zones_all:
            # Generate the contour for the current zone
            self.get_contour(superpos_zone)
            # Generate the centerline for the current zone
            self.get_centerline(superpos_zone)  
            # Generate patches along the centerline
            patches = self.patches_along_centerline()
            # Convert patches to polygons
            patches_poly = self.patch_to_polygon(patches)
            self.patches_list.append(patches)
            self.patches_poly_list.append(patches_poly)

        
    def get_contour(self, superpos_zone):
        """Get the contour of a single superposition zone"""
        contour_bulk = find_contours(superpos_zone.astype(int))[0]
        coords = approximate_polygon(contour_bulk, tolerance=self.tol)
        
        contour_x = coords[:, 1].astype(int)  # Column indices
        contour_y = coords[:, 0].astype(int)  # Row indices
        
        # Convert indices to Swiss coordinates
        contour = np.array([self.x_mesh[contour_y, contour_x] + 25 / 2, 
                                 self.y_mesh[contour_y, contour_x] - 25 / 2]).T
        self.contours_list.append(contour)

    
    def get_centerline(self, superpos_zone):
        """Get the centerline of the superposition zone using PCA."""
        mask_coords = np.column_stack(np.where(superpos_zone))

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
        
        self.centerlines_list.append(centerline)    
    
    def patches_along_centerline(self):
        patches = []
        # Convert last contour stored and centerline to Shapely polygon and LineString
        contour = self.contours_list[-1]
        centerline = self.centerlines_list[-1]
        
        contour_polygon = Polygon(contour)
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
                valid_start_found = True
                patches.append(patch)

            else:
                # Move start point forward
                start_dist += 20  # changer incrémentation?
                
        
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
            patches.append(patch)
            
            # Move to next position
            current_dist += self.sample_distance
            
        # check for last patch not to intersect the contour, remove it if it does 
        last_patch = patches[-1]
        last_patch_poly = Polygon(last_patch)

        if not last_patch_poly.within(contour_polygon):
            patches.pop()
            
        return patches         
        
    
    def create_patch(self, startpoint, direction, perp_direction, length, width):
        """A vérif car pas sur qu'il faille prendre le centre ???"""
        # Calculate half-dimensions
        half_width = width / 2
        
        corner1 = startpoint + length * direction + half_width * perp_direction
        corner2 = startpoint + length * direction - half_width * perp_direction
        corner3 = startpoint - half_width * perp_direction
        corner4 = startpoint + half_width * perp_direction
        
        # Create patch (closed polygon)
        patch = np.array([corner1, corner2, corner3, corner4, corner1])
        
        return patch
    
    def patch_to_polygon(self, patches):
        patches_poly = [Polygon(patch) for patch in patches]
        return patches_poly
        
    def compute_max_patch_length(self, idx):
        """Find the longest patch that fits along the centerline within the contour."""
        contour = self.contours_list[idx]
        centerline = self.centerlines_list[idx]
        
        contour_polygon = Polygon(contour)
        centerline_line = LineString(centerline)

        if len(centerline) < 2:
            raise ValueError("Centerline must have at least two points")

        direction = centerline[-1] - centerline[0]
        direction = direction / np.linalg.norm(direction)  # Normalize
        perp_direction = np.array([-direction[1], direction[0]])

        # Start from the first valid point. To do so, check first possible patch 
        valid_start_found = False
        start_dist = 0
        while not valid_start_found and start_dist < centerline_line.length:
            start_point = np.array(centerline_line.interpolate(start_dist).coords[0])
            # Generate random patch 
            patch = self.create_patch(start_point, direction, perp_direction, self.band_length, self.band_width)
            
            patch_poly = Polygon(patch)
            
            # Check if patch intersects with contour -> If so then the start position is valid
            if patch_poly.within(contour_polygon):
                valid_start_found = True
                
            else:
                start_dist += 20 

        # Then find the maximum length so the polygon remains into the contour  

        test_length = centerline_line.length  # Start with big length
        valid_end_found = False 

        while not valid_end_found:
            start_point = np.array(centerline_line.interpolate(start_dist).coords[0])
            patch = self.create_patch(start_point, direction, perp_direction, test_length, self.band_width)
            patch_poly = Polygon(patch)


            if not patch_poly.within(contour_polygon):
                test_length -= 10
            else:
                break  # Stop when the patch fits
            
        max_length = test_length
        self.max_patch_len.append(max_length)
        
        return start_point, max_length

    
    def create_single_patch(self, idx, start_point, length, width):
        """Generate a single patch covering the full length of the centerline."""
        contour = self.contours_list[idx]
        centerline = self.centerlines_list[idx]
        
        if len(centerline) < 2:
            return  # Not enough points to define a patch
        
        # Compute direction of the centerline (from start to end)
        direction = centerline[-1] - centerline[0]
        direction /= np.linalg.norm(direction)  # Normalize
        
        # Perpendicular direction for width
        perp_direction = np.array([-direction[1], direction[0]])    

        patch = self.create_patch(start_point, direction, perp_direction, length, width)
        
        return [patch] # l'enclure en nested list comme ça on a pas de soucis de dimension dans le plot du GUI
    

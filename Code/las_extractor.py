import laspy 
import numpy as np
from shapely.geometry import Point
import os
import time
import copy

class LasExtractor():
    def __init__(self, input, output, patch):
        self.input_file = input
        self.output_file = output
        self.patch_poly = patch # single patch
        
        self.las_reader()
        self.coords_extractions()
        self.patch_bounding_filtering()
        self.extract_las()
        
    def las_reader(self):
        self.las = laspy.read(self.input_file)
    
    def coords_extractions(self):
        self.coords = np.vstack((self.las.x, self.las.y)).transpose()
        
    def patch_bounding_filtering(self):
        
        min_x, min_y, max_x, max_y = self.patch_poly.bounds

        # Filtering the las X,Y coords with bounding box of patch 
        bbox_mask = ((self.coords[:, 0] >= min_x) & (self.coords[:, 0] <= max_x) & 
                    (self.coords[:, 1] >= min_y) & (self.coords[:, 1] <= max_y))

        points_in_bbox = self.coords[bbox_mask]
        self.coords_mask = np.zeros_like(bbox_mask) # mask to know which Point is in the Polygon
        # cette opération prend env. 6s
        self.coords_mask[bbox_mask] = [self.patch_poly.contains(Point(px, py)) for px, py in points_in_bbox]
        

    def copy_header(self):
        self.header = copy.deepcopy(self.las.header)
        filtered_point_count = np.sum(self.coords_mask)
        self.header.point_count = filtered_point_count  
        
    def extract_las(self):
        
        self.copy_header()
            # Create a new LAS file with the adjusted header
        new_las = laspy.LasData(self.header)    
        
        # Copy all point attributes while preserving IDs
        for dimension in self.las.point_format.dimension_names:
            data = getattr(self.las, dimension)
            setattr(new_las, dimension, data[self.coords_mask])

        
        new_las.write(self.output_file)
        
    
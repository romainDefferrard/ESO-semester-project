import laspy 
import numpy as np # type: ignore
from shapely.geometry import Point
from scipy.spatial import KDTree # type: ignore
from shapely.strtree import STRtree
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 
import os
import time
import copy
import subprocess
import pylas


class LasExtractor():
    def __init__(self, input, output, patch):
        self.input_file = input
        self.output_file = output
        self.patch_poly = patch # single patch
        
        self.las_reader()
        self.coords_extractions()
        self.patch_filtering_knn_classifier()
        self.extract_las()
        
    def las_reader(self):
        """LAS/LAZ reader depending on the format
        """
        with laspy.open(self.input_file) as fh:
            print('Points from Header:', fh.header.point_count)
            self.las = fh.read()


    def coords_extractions(self):
        self.coords = np.vstack((self.las.x, self.las.y)).transpose()
        print(self.coords)
        
    def patch_bounding_filtering(self):
        """Not used anymore, before knn method
        """
        min_x, min_y, max_x, max_y = self.patch_poly.bounds

        # Filtering the las X,Y coords with bounding box of patch 
        bbox_mask = ((self.coords[:, 0] >= min_x) & (self.coords[:, 0] <= max_x) & 
                    (self.coords[:, 1] >= min_y) & (self.coords[:, 1] <= max_y))

        points_in_bbox = self.coords[bbox_mask]
        self.coords_mask = np.zeros_like(bbox_mask) # mask to know which Point is in the Polygon
        # cette opération prend env. 6s
        self.coords_mask[bbox_mask] = [self.patch_poly.contains(Point(px, py)) for px, py in points_in_bbox]
        

    def patch_filtering_knn_classifier(self, k=5, prob_threshold=0.3, sample_size=40000):
        min_x, min_y, max_x, max_y = self.patch_poly.bounds
        
        # Filter points using bounding box
        bbox_mask = ((self.coords[:, 0] >= min_x) & (self.coords[:, 0] <= max_x) & 
                    (self.coords[:, 1] >= min_y) & (self.coords[:, 1] <= max_y))
        points_in_bbox = self.coords[bbox_mask]

        # Convert points to Pandas DataFrame
        df = pd.DataFrame(points_in_bbox, columns=['x', 'y'])
        # Sample a subset of points to manually label
        df_short = df.sample(n=min(sample_size, len(df)), random_state=42).copy()

        # Label points: 1 if inside the polygon, 0 otherwise
        df_short['labels'] = [self.patch_poly.contains(Point(px, py)) for px, py in zip(df_short.x, df_short.y)]

        # Train KNN classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(df_short[['x', 'y']], df_short['labels'])

        # Predict probabilities for all points
        df['predict'] = knn.predict_proba(df[['x', 'y']])[:, 1]  # Get probability of being inside

        # Filter points based on probability threshold
        filtered_df = df[df['predict'] > prob_threshold]

        # Update mask
        self.coords_mask = np.zeros(len(self.coords), dtype=bool)
        self.coords_mask[bbox_mask] = df['predict'].values > prob_threshold

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
        
    
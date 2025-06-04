"""
Filename: raster_loader.py
Author: Romain Defferrard
Date: 04-06-2025

Description:
    This module defines the RasterLoader class, which loads a buffered subsection of a raster (DTM)
    corresponding to the spatial extent of multiple flights. It uses rasterio to read data and
    produces both the raster values and a coordinate meshgrid for further processing.

    The main output is an instance providing:
        - self.raster: 2D numpy array of elevation values.
        - self.x_mesh, self.y_mesh: 2D arrays of Swiss projected coordinates (e.g., LV95 or LV03).
        - self.map_bounds: Buffered bounding box derived from flight area.
"""
import rasterio
import numpy as np
from typing import List


class RasterLoader:
    def __init__(self, config: dict, flight_bounds: List[float]) -> None:
        """
        Initializes RasterLoader and loads the raster.

        Input:
            config (dict): configuration dictionary.
                - DTM_PATH (str): Path to raster file.
                - RASTER_BUFFER (float): Buffer distance [m].
            flight_bounds (list[float]): [E_min, E_max, N_min, N_max] bounds of flight area.

        Output:
            None (but sets self.raster, self.x_mesh, self.y_mesh, self.map_bounds)
        """
        self.file_path = config["DTM_PATH"]
        self.buffer = config["RASTER_BUFFER"]
        self.flight_bounds = flight_bounds
        
        self.map_bounds = {}

        self.raster: np.ndarray
        self.x_mesh: np.ndarray
        self.y_mesh: np.ndarray

        self.compute_map_bounds() 
        self.load()

    def load(self) -> np.ndarray:
        """
        Loads the raster window corresponding to the buffered map bounds.

        Input:
            None

        Output:
            np.ndarray: The clipped raster values inside the buffered bounds.
        """
        with rasterio.open(self.file_path) as src:
            # Get raster resolution (pixel size in x and y)
            res_x, res_y = src.res
            # Ensure correct number of points in x/y
            x_coords = np.arange(self.map_bounds[0], self.map_bounds[1] + res_x, res_x)  # Add `+res_x`
            y_coords = np.arange(self.map_bounds[3], self.map_bounds[2] - res_y, -res_y)  # Add `-res_y`
            self.x_mesh, self.y_mesh = np.meshgrid(x_coords, y_coords)
            # Get the exact raster indices for slicing
            row_start, col_start = src.index(x_coords[0], y_coords[0])  # Top-left corner
            row_end, col_end = src.index(x_coords[-1], y_coords[-1])  # Bottom-right corner

            window = rasterio.windows.Window.from_slices(
                (row_start, row_end + 1), (col_start, col_end + 1)  # Add +1 to include last row  # Add +1 to include last column
            )

            self.raster = src.read(1, window=window)

            return self.raster

    def compute_map_bounds(self) -> None:
        """
        Computes a buffered bounding box around the flight area.

        Input:
            None

        Output:
            None (updates self.map_bounds as [E_min, E_max, N_min, N_max])
        """
        buffer_coef = np.array([-self.buffer, self.buffer, -self.buffer, self.buffer])
        bounds_array = np.array(self.flight_bounds)
        self.map_bounds = bounds_array + buffer_coef

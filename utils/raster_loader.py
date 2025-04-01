import rasterio
import numpy as np


class RasterLoader:
    def __init__(self, config, flight_bounds):
        self.file_path = config["MNT_PATH"]
        self.flight_bounds = flight_bounds

        self.map_bounds = {}

        self.compute_map_bounds() # still need to modify it 
        self.load()

    def load(self):
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

            # Create the rasterio window
            window = rasterio.windows.Window.from_slices(
                (row_start, row_end + 1), (col_start, col_end + 1)  # Add +1 to include last row  # Add +1 to include last column
            )

            self.raster = src.read(1, window=window)

            return self.raster

    def compute_map_bounds(self):
        # MODIFIER
        # get max diff flight/map
        # to compute FOV and add ± to bounds
        coef = np.array([-1000, 1000, -1000, 1000])
        bounds_array = np.array(self.flight_bounds)
        self.map_bounds = bounds_array + coef

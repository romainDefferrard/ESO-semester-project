"""
import rasterio
import numpy as np
from multiprocessing import Pool

class RasterLoader:
    def __init__(self, file_path, epsg, flight_bounds, tile_size=256, num_processes=4):
        self.file_path = file_path
        self.epsg = epsg
        self.flight_bounds = flight_bounds
        self.tile_size = tile_size  # Size of each tile (e.g., 256x256 pixels)
        self.num_processes = num_processes  # Number of processes to use for multiprocessing
        
        self.map_bounds = {}
        self.raster = None
        
    #    self.check_projection()
        self.compute_map_bounds()
        self.load()
        
    def load(self):
        # Get the number of tiles required for each dimension
        x_min, x_max, y_min, y_max = self.map_bounds
        n_cols = int((x_max - x_min) / self.tile_size)
        n_rows = int((y_max - y_min) / self.tile_size)
        
        # Create a pool of workers to load the tiles in parallel
        with Pool(self.num_processes) as pool:
            # Define the arguments for each tile (based on row and col)
            tile_indices = [(i, j, x_min, x_max, y_min, y_max) for i in range(n_cols) for j in range(n_rows)]
            
            # Parallelize the loading of each tile
            results = pool.starmap(self.load_tile, tile_indices)
        
        # Combine the results (rasters) into the final raster
        self.raster = np.vstack([np.hstack(results[i:i + n_cols]) for i in range(0, len(results), n_cols)])

    def load_tile(self, tile_idx, row, col, x_min, x_max, y_min, y_max):
        # Calculate the bounds of the current tile
        x_start = x_min + col * self.tile_size
        x_end = x_start + self.tile_size
        y_start = y_min + row * self.tile_size
        y_end = y_start + self.tile_size
        
        # Open the raster and read the data for the tile
        with rasterio.open(self.file_path) as src:
            
        
            # Get raster resolution (pixel size in x and y)
            res_x, res_y = src.res
            # Ensure correct number of points in x/y
            x_coords = np.arange(x_start, x_end + res_x, res_x)  # Add `+res_x`
            y_coords = np.arange(y_start, y_end - res_y, -res_y)  # Add `-res_y`
            self.x_mesh, self.y_mesh = np.meshgrid(x_coords, y_coords)
            # Get the exact raster indices for slicing
            row_start, col_start = src.index(x_coords[0], y_coords[0])  # Top-left corner
            row_end, col_end     = src.index(x_coords[-1], y_coords[-1])  # Bottom-right corner

            # Create the rasterio window
            window = rasterio.windows.Window.from_slices(
                (row_start, row_end + 1),  # Add +1 to include last row
                (col_start, col_end + 1)   # Add +1 to include last column
            )
            
            tile_data = src.read(1, window=window)
        
        return tile_data

    def check_projection(self):
        with rasterio.open(self.file_path) as src:
            if src.crs is None:
                raise ValueError(f"Raster has no defined coordinate system")
            
            # Check if the CRS matches
            if src.crs != rasterio.crs.CRS.from_epsg(self.epsg):
                current_epsg = src.crs.to_epsg() or "unknown"
                raise ValueError(f"Raster is not in EPSG:2056. Current CRS: EPSG:{current_epsg}")
            
            return True

    def compute_map_bounds(self):
        # Adjust the map bounds based on the flight_bounds and add some margin (±250 meters)
        coef = np.array([-250, 250, -250, 250])
        bounds_array = np.array(self.flight_bounds)
        self.map_bounds = bounds_array + coef





   
    """
import rasterio
import numpy as np

class RasterLoader:
    def __init__(self, file_path, epsg, flight_bounds):
        self.file_path = file_path
        self.raster = None
        
        self.epsg = epsg
        self.flight_bounds = flight_bounds
        
        self.tile_size = 256
        
        
        self.map_bounds = {}

        
        self.check_projection()
        self.compute_map_bounds()
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
            row_end, col_end     = src.index(x_coords[-1], y_coords[-1])  # Bottom-right corner

            # Create the rasterio window
            window = rasterio.windows.Window.from_slices(
                (row_start, row_end + 1),  # Add +1 to include last row
                (col_start, col_end + 1)   # Add +1 to include last column
            )

            self.raster = src.read(1, window=window)

            return self.raster
    
    def check_projection(self):
        with rasterio.open(self.file_path) as src:
            if src.crs is None:
                raise ValueError(f"Raster has no defined coordinate system")
            
            # Check is both CRS are consistent 
            if src.crs != rasterio.crs.CRS.from_epsg(self.epsg):
                current_epsg = src.crs.to_epsg() or "unknown"
                raise ValueError(f"Raster is not in EPSG:2056. Current CRS: EPSG:{current_epsg}")
            
            return True
    
    def compute_map_bounds(self):
        # MODIFIER 
        # get max diff flight/map 
        # to compute FOV and add ± to bounds 
        coef = np.array([-1000, 1000, -1000, 1000])
        bounds_array = np.array(self.flight_bounds)
        self.map_bounds = bounds_array + coef



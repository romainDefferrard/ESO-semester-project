""" 
Filename: patch_generator.py
Author: Romain Defferrard
Date: 04-06-2025

Description:
    This module defines the PatchGenerator class, which generates rectangular patch polygons
    within overlapping LiDAR flight zones. It uses the raster mesh as the Swiss coordinate grid, 
    apply PCA to find the dominant direction of the estimated overlap, and sample patches along that centerline.

    The generated patches are stored as Patch objects with geometric metadata (center, direction, etc.).
"""
import numpy as np
from typing import List, Tuple, Optional
from skimage.measure import find_contours, approximate_polygon
from shapely.geometry import Polygon, LineString
from sklearn.decomposition import PCA
import warnings

from .patch_model import PatchParams, Patch


class PatchGenerator:
    def __init__(
        self, superpos_zones: List[np.ndarray], raster_mesh: Tuple[np.ndarray, np.ndarray], patch_params: Tuple[float, float, float]
    ):
        """
        Initializes the PatchGenerator class.

        Inputs:
            superpos_zones (List[np.ndarray]):           List of overlapping area masks.
            raster_mesh (Tuple[np.ndarray, np.ndarray]): (x_mesh, y_mesh) for spatial mapping.
            patch_params (Tuple[float, float, float]):   (length, width, sample_distance) of patches.
            
        Output:
            None
        """

        self.superpos_zones_all = superpos_zones
        self.x_mesh, self.y_mesh = raster_mesh
        self.band_length, self.band_width, self.sample_distance = patch_params

        self.tol = 0.05  # tolerance parameter for the contour generation
        self.contour = None
        self.patch_id = 1

        warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely.predicates")

        # Output
        self.patches_list = []
        self.centerlines_list = []
        self.contours_list = []
        self.max_patch_len = []
        self.patches_poly_list = []

        # Process all superposition zones
        self.process_zones()

    def process_zones(self) -> None:
        """
        Iterate through each overlapping zone to generate the contour polygon, PCA centerline and 
        series of rectangular patches.

         Stores outputs in:
            - self.contours_list (List[np.ndarray]):    polygon in Swiss coordinates
            - self.centerlines_list (List[np.ndarray]): PCA of overlaps 
            - self.patches_list (List[List[Patch]]):    list of patch objects per zone

        """
        for superpos_zone in self.superpos_zones_all:
            self.get_contour(superpos_zone)  # Generate the contour for the current zone
            self.get_centerline(superpos_zone)  # Generate the centerline for the current zone
            patches = self.patches_along_centerline()  # Generate patches along the centerline
            self.patches_list.append(patches)

    def get_contour(self, superpos_zone: np.ndarray) -> None:
        """
        Extract a polygonal contour from a superposition zone using skimage library. 

        Converts pixel indices to Swiss coordinate reference using mesh grids and stores
        the result in self.contours_list.

        Input:
            superpos_zone (np.ndarray): Single boolean mask of overlapping area

        Output:
            None
        """
        contour_bulk = find_contours(superpos_zone.astype(int))[0]
        coords = approximate_polygon(contour_bulk, tolerance=self.tol)

        contour_x = coords[:, 1].astype(int)  # Column indices
        contour_y = coords[:, 0].astype(int)  # Row indices

        # Convert indices to Swiss coordinates frame
        contour = np.array(
                        [self.x_mesh[contour_y, contour_x] + 25 / 2, self.y_mesh[contour_y, contour_x] - 25 / 2]
                        ).T  # Add 25/2 to get the center of the pixel
        self.contours_list.append(contour)

    def get_centerline(self, superpos_zone: np.ndarray) -> None:
        """
        Estimate the main orientation of a superposition zone using PCA on its pixel coordinates.

        Input:
            superpos_zone (np.ndarray): Single boolean mask of overlapping area

        Output:
            None
        """
        mask_coords = np.column_stack(np.where(superpos_zone))
        coord_points = np.array([self.x_mesh[mask_coords[:, 0], mask_coords[:, 1]], self.y_mesh[mask_coords[:, 0], mask_coords[:, 1]]]).T

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
        centerline = pca.inverse_transform(line_points)

        self.centerlines_list.append(centerline)

    def patches_along_centerline(self) -> List[np.ndarray]:
        """
        Generates rectangular patches at regular intervals along the PCA centerline.

        Patches are defined based on configured band length/width and spacing.
        A patch is only retained if fully within the previously computed contour polygon.

        Returns:
            List[np.ndarray]: List of valid Patch objects
        """
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
        valid_end_found = False

        start_dist = 0

        while not valid_start_found and start_dist < centerline_line.length:

            # Get point at current distance along the centerline
            start_point = np.array(centerline_line.interpolate(start_dist).coords[0])
            params = PatchParams(start_point, direction, perp_direction, self.band_length, self.band_width)
            patch = self.create_patch(params)
            patch_poly = patch.shapely_polygon  # in the Patch class

            if patch_poly.within(contour_polygon):
                valid_start_found = True
            else:
                start_dist += 100  # changer incrémentation plus grande nécessaire sinon effet de bord trop importants

        patches.append(patch)
        self.patch_id += 1

        if not valid_start_found:
            return []

        # Generate subsequent patches
        current_dist = start_dist + self.sample_distance
        while current_dist < centerline_line.length:
            current_point = np.array(centerline_line.interpolate(current_dist).coords[0])
            params = PatchParams(current_point, direction, perp_direction, self.band_length, self.band_width)
            patch = self.create_patch(params)
            patches.append(patch)
            self.patch_id += 1
            current_dist += self.sample_distance

        # check for last patch not to intersect the contour, remove it if it does
        while not valid_end_found:
            last_patch = patches[-1]
            last_patch_poly = last_patch.shapely_polygon
            if not last_patch_poly.within(contour_polygon):
                patches.pop()
            else:
                valid_end_found = True

        return patches

    def create_patch(self, params: PatchParams) -> np.ndarray:
        """
        Create a rectangular patch polygon from its geometric definition (start point,
        direction, perpendicular direction, width and length).

        Converts the 4 corners to a polygon and stores both geometry and metadata.

        Input:
            params (PatchParams): Geometric parameters of the patch

        Returns:
            patch: Patch object with polygon and metadata
        """
        half_width = params.width / 2

        corner1 = params.startpoint + params.length * params.direction + half_width * params.perp_direction
        corner2 = params.startpoint + params.length * params.direction - half_width * params.perp_direction
        corner3 = params.startpoint - half_width * params.perp_direction
        corner4 = params.startpoint + half_width * params.perp_direction

        corners = np.array([corner1, corner2, corner3, corner4, corner1])
        polygon = Polygon(corners)

        center = params.startpoint + params.length / 2 * params.direction
        direction = params.direction
        perp_direction = params.perp_direction

        patch = Patch(
            id=self.patch_id,
            patch_array=corners,
            shapely_polygon=polygon,
            metadata={"center": center, "direction": direction, "perp_direction": perp_direction, "length": params.length, "width": params.width},
        )

        # patch = Patch(id=self.patch_id, patch_array=corners, shapely_polygon=polygon)
        return patch

    def compute_max_patch_length(self, idx: int) -> Tuple[np.ndarray, float]:
        """
        Compute the longest valid patch length that remains entirely within the contour. (Called in gui.py)

        This function starts from the first valid position along the centerline and 
        gradually reduces the patch length until the generated patch is fully contained 
        within the corresponding superposition contour.

        Input:
            idx (int): Index of the superposition zone (referring to contours_list and centerlines_list)

        Output:
            - start_point (np.ndarray): Coordinates of the starting point for the patch
            - max_length (float):       Maximum patch length that does not exceed the contour bounds
        """
        self.patch_id += 0

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
            params = PatchParams(start_point, direction, perp_direction, self.band_length, self.band_width)
            patch = self.create_patch(params)
            patch_poly = patch.shapely_polygon

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
            params = PatchParams(start_point, direction, perp_direction, test_length, self.band_width)
            patch = self.create_patch(params)
            patch_poly = patch.shapely_polygon

            if not patch_poly.within(contour_polygon):
                test_length -= 10
            else:
                break  # Stop when the patch fits

        max_length = test_length
        self.max_patch_len.append(max_length)

        return start_point, max_length

    def create_single_patch(self, idx: int, start_point: np.ndarray, length: float, width: float) -> Optional[List[np.ndarray]]:
        """
        Generate a single rectangular patch of given dimensions, placed at a specified start point
        along the centerline of the selected superposition zone. (Called in gui.py)

        Input:
            idx (int): Index of the zone (from centerlines_list)
            start_point (np.ndarray): Starting point of the patch in (x, y) coordinates
            length (float): Length of the patch along the main axis (centerline direction)
            width (float): Width of the patch perpendicular to the main axis

        Output:
            Optional[List[np.ndarray]]:
                - A list containing a single Patch object, or None if centerline is invalid
        """
        centerline = self.centerlines_list[idx]

        if len(centerline) < 2:
            return  # Not enough points to define a patch

        # Compute direction of the centerline (from start to end)
        direction = centerline[-1] - centerline[0]
        direction /= np.linalg.norm(direction)  # Normalize

        # Perpendicular direction for width
        perp_direction = np.array([-direction[1], direction[0]])

        params = PatchParams(start_point, direction, perp_direction, length, width)
        patch = self.create_patch(params)
        self.patch_id += 1

        return [patch]  # l'enclure en nested list comme ça on a pas de soucis de dimension dans le plot du GUI

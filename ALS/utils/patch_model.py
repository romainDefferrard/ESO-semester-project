"""
Filename: patch_model.py
Author: Romain Defferrard
Date: 04-06-2025
"""

from dataclasses import dataclass
import numpy as np
from shapely.geometry import Polygon



@dataclass
class PatchParams:
    """
    Stores the geometric parameters required to construct a rectangular patch.

    Attributes:
        startpoint (np.ndarray): Starting point (x, y) in Swiss coordinates of the patch along the centerline.
        direction (np.ndarray): Unit vector along the patch's main axis (length direction).
        perp_direction (np.ndarray): Unit vector perpendicular to the main axis (width direction).
        length (float): Length of the patch along the main direction [m].
        width (float): Width of the patch across the perpendicular direction [m].
    """
    startpoint: np.ndarray
    direction: np.ndarray
    perp_direction: np.ndarray
    length: float
    width: float

@dataclass
class Patch:
    """
    Represents a rectangular patch defined in 2D space.

    Attributes:
        id (int): Unique identifier for the patch on the scale of the acquisition mission.
        patch_array (np.ndarray): Coordinates of the patch corners (x, y) in Swiss coordinates.
        shapely_polygon (Polygon): Shapely Polygon object representing the patch geometry.
        metadata (dict): Additional information, such as center, direction, and dimensions.
    """
    id: int
    patch_array: np.ndarray
    shapely_polygon: Polygon
    metadata: dict
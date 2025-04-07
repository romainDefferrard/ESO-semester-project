from dataclasses import dataclass
import numpy as np
from shapely.geometry import Polygon



@dataclass
class PatchParams:
    startpoint: np.ndarray
    direction: np.ndarray
    perp_direction: np.ndarray
    length: float
    width: float

@dataclass
class Patch:
    id: int
    patch_array: np.ndarray
    shapely_polygon: Polygon
    metadata: dict

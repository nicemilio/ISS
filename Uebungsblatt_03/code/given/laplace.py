import numpy as np
import cv2

from typing import Literal

LAPLACE_L4 = None # TODO
LAPLACE_L8 = None # TODO

def linear_stretch_to_u8(img: np.ndarray) -> np.ndarray:
    # TODO
    return None

def laplace_response(gray_f32: np.ndarray, variant: Literal["L4", "L8"] = "L4") -> np.ndarray:
    # TODO
    return None

def zero_crossing(resp: np.ndarray, connectivity: Literal[4, 8] = 4, eps: float=1e-6) -> np.ndarray:
    # TODO
    return None

def laplace_edges(gray_f32: np.ndarray, variant: Literal["L4", "L8"] = "L4", connectivity: Literal[4, 8] = 4, eps: float=1e-6) -> np.ndarray:
    # TODO
    return None

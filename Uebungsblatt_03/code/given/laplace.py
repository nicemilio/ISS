import numpy as np
import cv2

from typing import Literal

LAPLACE_L4 = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)

LAPLACE_L8 = np.array([[1, 1, 1],
                       [1, -8, 1],
                       [1, 1, 1]], dtype=np.float32)

def linear_stretch_to_u8(img: np.ndarray) -> np.ndarray:
    min_val, max_val = np.min(img), np.max(img)
    stretched = (img - min_val) / (max_val - min_val + 1e-8) * 255.0
    return stretched.astype(np.uint8)

def laplace_response(gray_f32: np.ndarray, variant: Literal["L4", "L8"] = "L4") -> np.ndarray:
    kernel = LAPLACE_L4 if variant == "L4" else LAPLACE_L8
    resp = cv2.filter2D(gray_f32, cv2.CV_32F, kernel)
    return resp

def zero_crossing(resp: np.ndarray, connectivity: Literal[4, 8] = 4, eps: float = 1e-6) -> np.ndarray:
    h, w = resp.shape
    edges = np.zeros((h, w), dtype=np.float32)
    
    # Nachbarschaft definieren
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:  # 8er Nachbarschaft
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            center = resp[y, x]
            for dy, dx in neighbors:
                neighbor = resp[y + dy, x + dx]
                # Wenn Vorzeichenwechsel und Betrag groß genug
                if (center * neighbor < 0) and (abs(center - neighbor) > eps):
                    edges[y, x] = abs(center)
                    break
    return edges

def laplace_edges(gray_f32: np.ndarray, variant: Literal["L4", "L8"] = "L4",
                  connectivity: Literal[4, 8] = 4, eps: float = 1e-6) -> np.ndarray:
    resp = laplace_response(gray_f32, variant)
    edges = zero_crossing(resp, connectivity, eps)
    stretched = linear_stretch_to_u8(edges)
    return stretched
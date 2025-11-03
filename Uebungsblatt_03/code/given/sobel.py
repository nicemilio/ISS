import numpy as np
import cv2
from scipy.signal import convolve2d

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

SOBEL_Y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

def linear_stretch_to_u8(img: np.ndarray) -> np.ndarray:
    # Minimum und Maximum des Bildes bestimmen
    min_val = np.min(img)
    max_val = np.max(img)

    # Schutz vor Division durch 0 (falls konstantes Bild)
    if max_val == min_val:
        return np.zeros_like(img, dtype=np.uint8)

    # Lineare Skalierung auf [0, 255]
    stretched = (img - min_val) / (max_val - min_val) * 255.0

    # In uint8 konvertieren
    return stretched.astype(np.uint8)

def sobel_gradients(gray_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    # 2D-Faltung (Rand mit Spiegelung behandeln)
    Gx = convolve2d(gray_f32, SOBEL_X, mode="same", boundary="symm")
    Gy = convolve2d(gray_f32, SOBEL_Y, mode="same", boundary="symm")

    return Gx, Gy

def magnitude_and_orientation(gx: np.ndarray, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Betrag nach euklidischer Norm
    magnitude = np.sqrt(gx**2 + gy**2)

    # Orientierung mit atan2 (liefert Winkel zwischen -π und π)
    orientation = np.arctan2(gy, gx)

    return magnitude, orientation
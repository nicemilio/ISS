import numpy as np
import cv2
from scipy.signal import convolve2d

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

SOBEL_Y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

# Skaliert ein Bild linear auf den Bereich [0, 255] und konvertiert es zu uint8
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

#Liefert Kanten in x- und y-Richtung
def sobel_gradients(gray_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    # 2D-Faltung mit gleichen Dimensionen(Rand mit Spiegelung behandeln)
    Gx = convolve2d(gray_f32, SOBEL_X, mode="same", boundary="symm") # vertikale Kanten
    Gy = convolve2d(gray_f32, SOBEL_Y, mode="same", boundary="symm") # horizontale Kanten

    return Gx, Gy

# Liefert Betrag und Orientierung der Kanten
def magnitude_and_orientation(gx: np.ndarray, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Betrag nach euklidischer Norm
    magnitude = np.sqrt(gx**2 + gy**2) # Kantenstärke

    # Orientierung mit atan2 (liefert Winkel zwischen -π und π)
    orientation = np.arctan2(gy, gx) # Kantenrichtung

    return magnitude, orientation
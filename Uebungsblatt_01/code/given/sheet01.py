import cv2
import numpy as np

from os.path import *

def exercise2a(a: int, k: int, image_folder="."):

    image = cv2.imread(join(image_folder, "Lena_512x512.png"))

    # In float32 umwandeln, damit Multiplikation keine Überläufe verursacht
    image_float = image.astype(np.float32)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            r = image_float[y, x, 2]  # Rot-Kanal
            g = image_float[y, x, 1]  # Grün-Kanal
            b = image_float[y, x, 0] * a  # Blau-Kanal multiplizieren
            b = min(b, 255)  # Werte auf 255 begrenzen

            # Zurück in int umwandeln
            image[y, x] = (np.uint8(b), np.uint8(g), np.uint8(r))

    cv2.imshow("Ex. 2a float", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def exercise2b(a: int, k: int, image_folder="."):

    image = cv2.imread(join(image_folder, "Lena_512x512.png"))

    # In float32 umwandeln, damit Multiplikation overflow-sicher ist
    image_float = image.astype(np.float32)

    #Blau-Kanalverstärken
    image_float[:, :, 0] = np.clip(image_float[:, :, 0] * a, 0, 255)

    # Wieder in uint8 zurückkonvertieren
    image = image_float.astype(np.uint8)

    # Bild anzeigen
    cv2.imshow("Ex. 2b)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def exercise3(image_folder="."):
    
    image = cv2.imread(join(image_folder, "Lena_512x512.png"))
    mask = cv2.imread(join(image_folder, "Maske_Lena_512x512.png"))

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):

            if not mask[y, x].any():
                image[y, x] = (0, 0, 0) 

    cv2.imshow("Ex. 3)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # ------------------
    # --- EXERCISE 2 ---
    # ------------------

    a = 5 # scale factor
    k = 2 # channel

    exercise2a(a=a, k=k)
    exercise2b(a=a, k=k)

    # ------------------
    # --- EXERCISE 3 ---
    # ------------------

    exercise3()

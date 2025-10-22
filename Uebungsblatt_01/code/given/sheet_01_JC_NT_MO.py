import cv2
import numpy as np

from os.path import *
 # OpenCV + NumPy utilities; OpenCV uses BGR channel order by default

def exercise2a(a: int, k: int, image_folder="."):
    # Exercise 2a: Scale the blue channel per pixel using overflow-safe math

    image = cv2.imread(join(image_folder, "Lena_512x512.png"))
    # Read image as BGR uint8 array of shape (H, W, 3)

    # In float32 umwandeln, damit Multiplikation keine Überläufe verursacht
    image_float = image.astype(np.float32)

    # Schleife über alle Pixel (langsamer, aber didaktisch nachvollziehbar)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            r = image_float[y, x, 2]  # Rot-Kanal
            g = image_float[y, x, 1]  # Grün-Kanal
            b = image_float[y, x, 0] * a  # Blau-Kanal multiplizieren
            b = min(b, 255)  # Werte auf 255 begrenzen

            # Zurück in int umwandeln
            image[y, x] = (np.uint8(b), np.uint8(g), np.uint8(r))

    # Bild anzeigen
    cv2.imshow("Ex. 2a float", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def exercise2b(a: int, k: int, image_folder="."):
    # Exercise 2b: Vektorisierte (schnelle) Variante für den Blau-Kanal

    image = cv2.imread(join(image_folder, "Lena_512x512.png"))
    # Bild ist BGR; Kanalindex 0 entspricht Blau

    # In float32 umwandeln, damit Multiplikation overflow-sicher ist
    image_float = image.astype(np.float32)

    # Blau-Kanal verstärken und auf gültigen Bereich [0, 255] begrenzen
    image_float[:, :, 0] = np.clip(image_float[:, :, 0] * a, 0, 255)

    # Wieder in uint8 zurückkonvertieren
    image = image_float.astype(np.uint8)

    # Bild anzeigen
    cv2.imshow("Ex. 2b)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def exercise3(image_folder="."):
    # Exercise 3: Maske anwenden – Pixel außerhalb der Maske werden geschwärzt
    
    image = cv2.imread(join(image_folder, "Lena_512x512.png"))
    mask = cv2.imread(join(image_folder, "Maske_Lena_512x512.png"))
    # Erwartet Maske gleicher Größe wie Bild; nicht-schwarze Maskenpixel lassen Bild unverändert

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # mask[y, x].any() ist True, wenn irgendein Kanal != 0 ist (also kein reines Schwarz)
            if not mask[y, x].any():  # Nur rein schwarze Maskenpixel setzen das Bildpixel auf Schwarz
                image[y, x] = (0, 0, 0) 

    cv2.imshow("Ex. 3)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # ------------------
    # --- EXERCISE 2 ---
    # ------------------

    a = 5 # scale factor (z.B. 5x Blau verstärken)
    k = 2 # channel (nicht benutzt in diesem Beispiel)

    exercise2a(a=a, k=k)
    exercise2b(a=a, k=k)

    # ------------------
    # --- EXERCISE 3 ---
    # ------------------

    exercise3()

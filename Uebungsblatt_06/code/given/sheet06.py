import argparse

import cv2
from os.path import *
import numpy as np


def exercise1(image_folder=".", input: str = None):
    image_path = join(image_folder, input)
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError:
        print("Fehler beim Laden des Bildes. Bitte Pfad und Dateinamen prüfen.")
        return

    # Todo Ihre Lösung
    #  Achten sie darauf, den Dijkstra Algorithmus wie in der Vorlesung beschrieben (Folien 38ff.) zu implementieren.
    
def show_image(title, image, info_text=None):
    """
    Zeigt ein Bild in einem Fenster an und wartet auf einen Tastendruck.
    Konvertiert Float-Bilder (z.B. von Sobel) korrekt für die Anzeige.
    NEU: Fügt optional Parameter-Infos (info_text) zum Fenstertitel hinzu.
    """
    # Fenstertitel erstellen
    window_title = title
    if info_text:
        # Füge die Parameter-Infos zum Titel hinzu
        window_title = f"{title} | {info_text}"
    
    # Für die Anzeige von Float-Bildern (wie Sobel-Ergebnis) normalisieren
    if image.dtype == np.float64:
        # Normalisiere das Bild auf den Bereich 0-255
        image_display = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        # 8-Bit-Bilder (wie das Original oder Canny-Ergebnis)
        image_display = image
    
    cv2.imshow(window_title, image_display)
    # Gib den vollen Titel auch in der Konsole aus
    print(f"'{window_title}' wird angezeigt. Drücke eine beliebige Taste im Bildfenster, um fortzufahren...")
    cv2.waitKey(0) # Warte unendlich lange auf eine Taste

# --- Canny-Algorithmus Schritte (AKTUALISIERT mit Parametern) ---

def apply_gaussian(image, kernel_size=(5, 5), sigma=0):
    """Schritt 1: Rauschunterdrückung mit Gauß-Filter"""
    # (5, 5) ist die Kernel-Größe, 0 ist die Standardabweichung
    return cv2.GaussianBlur(image, kernel_size, sigma)

def apply_sobel_magnitude(image, ksize=3):
    """Schritt 2: Gradientenstärke (Magnitude) finden"""
    # Berechne Gradienten in X- und Y-Richtung
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    
    # Berechne die absolute Magnitude (Stärke)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    
    return magnitude

def apply_canny_final(image, low_threshold=100, high_threshold=200):
    """
    Schritt 3 & 4: Non-Maximum Suppression und Hysterese-Thresholding.
    """
    return cv2.Canny(image, low_threshold, high_threshold)

# --- Hauptfunktion (AKTUALISIERT mit Parameter-Übergabe) ---

def exercise3(image_folder="."):
    try:
        image_path = join(image_folder, "Testbild_Lena_512x512.png")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise FileNotFoundError(f"Bild nicht gefunden unter: {image_path}")

        print("Starte Canny-Edge-Detection Schritt für Schritt...")
        
        # Originalbild anzeigen
        show_image("Originalbild", image)

        # --- Schritt 1: Rauschunterdrückung ---
        # Definiere die Parameter hier
        gauss_kernel = (5, 5)
        gauss_sigma = 0  # 0 bedeutet, OpenCV berechnet Sigma aus der Kernel-Größe
        
        blurred_image = apply_gaussian(image, gauss_kernel, gauss_sigma)
        
        # Erstelle den Info-String für die Anzeige
        info_str_1 = f"Kernel={gauss_kernel}, Sigma={gauss_sigma} (auto)"
        show_image("Schritt 1: Gaussian Blur", blurred_image, info_text=info_str_1)

        # --- Schritt 2: Gradienten-Magnitude ---
        sobel_ksize = 3
        
        gradient_magnitude = apply_sobel_magnitude(blurred_image, ksize=sobel_ksize)
        
        info_str_2 = f"Sobel Kernel Size={sobel_ksize}"
        show_image("Schritt 2: Gradienten-Magnitude (Sobel)", gradient_magnitude, info_text=info_str_2)

        # --- Schritt 3 & 4: NMS + Hysterese (Das 'echte' Canny-Ergebnis) ---
        # Definiere die Schwellenwerte (t1 und t2)
        t_low = 100  # t1
        t_high = 200 # t2
        
        final_edges = apply_canny_final(image, t_low, t_high)
        
        info_str_3 = f"T_low (t1)={t_low}, T_high (t2)={t_high}"
        show_image("Schritt 3+4: Endergebnis (cv2.Canny)", final_edges, info_text=info_str_3)
        
        print("Alle Schritte abgeschlossen. Schließe alle Fenster.")

    except FileNotFoundError as e:
        print(e)
        print("Fehler beim Laden des Bildes. Bitte Pfad und Dateinamen prüfen.")
        return
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
    finally:
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Intelligent Scissors")
    parser.add_argument("--input", type=str, default="Testbild_Gangman_300x200.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    image_folder = "."

    # ------------------
    # --- EXERCISE 1 ---
    # ------------------

    exercise1(image_folder=image_folder, input=args.input)


    # ------------------
    # --- EXERCISE 3 ---
    # ------------------

    exercise3(image_folder=image_folder)




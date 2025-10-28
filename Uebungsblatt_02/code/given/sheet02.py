import cv2
import numpy as np

from os.path import *

def exercise1(image_folder=".", resize_factor=6):

    image = cv2.imread(join(image_folder, "Weeki_Wachee_spring_10079u.png"))

    if image is None:
        print("Fehler beim Laden des Bildes. Bitte Pfad und Dateinamen prüfen.")
        return

    gamma = float(input("Bitte geben Sie den Wert für gamma ein (z.B. 0.5 oder 2.2): "))

    # apply gamma correction
    gamma_corrected = image.copy()

    # TODO: Ihre Lösung
    # ...

    image_out = cv2.resize(src=gamma_corrected, dsize=(
        gamma_corrected.shape[1]//resize_factor, gamma_corrected.shape[0]//resize_factor))
    
    image_in = cv2.resize(src=image, dsize=(
        image.shape[1]//resize_factor, image.shape[0]//resize_factor))

    cv2.imshow("Ex. 1)", np.hstack([image_in, image_out]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def filter_image(image: np.ndarray, kernel: np.ndarray):
    
    filtered_image = image.copy()

    height, width = filtered_image.shape[:2]
    
    output = np.zeros((height, width), dtype=np.float32)

    filtered_image = None

    # TODO: Ihre Lösung
    # ...
    
    return filtered_image

if __name__ == "__main__":
    
    image_folder = "code/given/"

    # ------------------
    # --- EXERCISE 1 ---
    # ------------------

    exercise1(image_folder=image_folder)

    # ------------------
    # --- EXERCISE 5 ---
    # ------------------

    input_image = cv2.imread(join(image_folder, "Testbild_Werkzeuge_768x576.png"))

    kernel_33 = None # TODO: Ihre Lösung
    result = filter_image(image=input_image, kernel=kernel_33)
    
    cv2.imshow("Ex. 5) 3x3 Mittelwertfilter", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    kernel_55 = None # TODO: Ihre Lösung
    result = filter_image(image=input_image, kernel=kernel_55)
    
    cv2.imshow("Ex. 5) 5x5 Mittelwertfilter", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

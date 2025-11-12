import cv2
import numpy as np
from os.path import *


def median_filter(img: np.ndarray, mask: int = 3) -> np.ndarray:

    if mask is None or mask < 1 or mask % 2 == 0:
        mask = 3
    if img.ndim == 2:
        return cv2.medianBlur(img, mask)
    elif img.ndim == 3:
        channels = cv2.split(img)
        filtered = [cv2.medianBlur(ch, mask) for ch in channels]
        return cv2.merge(filtered)
    else:
        return img

def diffusion_filter(
    i: np.ndarray,
    eps0: float = 1.0,
    iters: int = 500,
    lambda_: float = 1) -> np.ndarray:
    # TODO: Ihre Lösung
    # ...
    return i

def lin_scaling(img_diff: np.ndarray, img_median: np.ndarray) -> np.ndarray:
    # TODO: Ihre Lösung
    # ...
    lin_scaled_image = None
    return lin_scaled_image


def exercise1(image_folder="."):
    try:
        image = cv2.imread(join(image_folder, "Testbild_Lena_512x512.png"), cv2.IMREAD_COLOR)
        image_2 = cv2.imread(join(image_folder, "Testbild_Rauschen_640x480.png"), cv2.IMREAD_GRAYSCALE)

    except FileNotFoundError:
        print("Fehler beim Laden des Bildes. Bitte Pfad und Dateinamen prüfen.")
        return


    mask = 3 
    mask_2 = 5 

    out_img = median_filter(image, mask=mask)
    out_img_2 = median_filter(image_2, mask=mask_2)



    cv2.imshow("Ex. 1) Lena", out_img)
    cv2.imshow("Ex. 1) Rauschen", out_img_2)
    cv2.waitKey(0)

    saved_1 = cv2.imwrite("ex1_lena.png", out_img)
    saved_2 = cv2.imwrite("ex1_rauschen.png", out_img_2)
    print("Gespeichert:", saved_1, saved_2)
    cv2.destroyAllWindows()


def exercise2(image_folder="."):
    try:
        image = cv2.imread(join(image_folder, "Testbild_Rauschen_640x480.png"), cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError:
        print("Fehler beim Laden des Bildes. Bitte Pfad und Dateinamen prüfen.")
        return
    img_arr = np.asarray(image, dtype=np.float32) / 255.0
    out_img = diffusion_filter(i=img_arr, eps0=1.0, iters=500, lambda_=1)
    cv2.imshow("Ex. 2) Rauschen", out_img)
    cv2.waitKey(0)
    saved_1 = cv2.imwrite("ex2_rauschen_.png", out_img)
    print("Gespeichert:", saved_1)
    cv2.destroyAllWindows()


def exercise3(image_folder="."):
    try:
        image_diffusion = cv2.imread(join(image_folder, "Testbild_Rauschen_640x480_Diffusion.png"), cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError:
        print("Fehler beim Laden des Bildes. Bitte Pfad und Dateinamen prüfen.")
        return
    try:
        image_median = cv2.imread(join(image_folder, "Testbild_Rauschen_640x480_Median.png"), cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError:
        print("Fehler beim Laden des Bildes. Bitte Pfad und Dateinamen prüfen.")
        return


    out_img = lin_scaling(image_diffusion, image_median)
    cv2.imshow("Ex. 3) Lineare Streckung", out_img)
    cv2.waitKey(0)
    saved_1 = cv2.imwrite("ex3_streckung.png", out_img)
    print("Gespeichert:", saved_1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    source_images = "."

    # ------------------
    # --- EXERCISE 1 ---
    # ------------------

    exercise1(image_folder=source_images)

    # ------------------
    # --- EXERCISE 2 ---
    # ------------------

    exercise2(image_folder=source_images)

    # ------------------
    # --- EXERCISE 3 ---
    # ------------------

    exercise3(image_folder=source_images)

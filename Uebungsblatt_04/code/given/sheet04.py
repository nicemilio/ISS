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

    # Sicherstellen, dass wir in float32 und [0,1] arbeiten
    I = i.astype(np.float32)
    if I.max() > 1.0:
        I = I / 255.0

    for _ in range(max(1, int(iters))):
        # A) Gradienten (Vorwärtsdifferenzen)
        Ix = np.zeros_like(I, dtype=np.float32)
        Iy = np.zeros_like(I, dtype=np.float32)
        Ix[:, :-1] = I[:, 1:] - I[:, :-1]   # ∂I/∂x
        Iy[:-1, :] = I[1:, :] - I[:-1, :]   # ∂I/∂y

        grad_mag = np.sqrt(Ix ** 2 + Iy ** 2)

        # B) Isotroper inhomogener Diffusionstensor D(|∇I|)
        D = 1.0 / (1.0 + (grad_mag / float(eps0)) ** 2)

        # Fluss j = -D ∇I
        jx = -D * Ix
        jy = -D * Iy

        # C) Gradienten des Flusses -> D) Divergenz
        div = np.zeros_like(I, dtype=np.float32)
        # ∂jx/∂x (Rückwärtsdifferenzen)
        div[:, 1:] += jx[:, 1:] - jx[:, :-1]
        # ∂jy/∂y (Rückwärtsdifferenzen)
        div[1:, :] += jy[1:, :] - jy[:-1, :]

        # E) Update
        I = I - float(lambda_) * div
        # Wertebereich stabilisieren
        I = np.clip(I, 0.0, 1.0)

    # Ausgabe als 8-bit Bild für OpenCV Anzeige/Speichern
    return (I * 255.0).astype(np.uint8)

def lin_scaling(img_diff: np.ndarray, img_median: np.ndarray) -> np.ndarray:
    # Differenz bilden (Median - Diffusion)
    diff = img_median.astype(np.float32) - img_diff.astype(np.float32)

    # Minimum und Maximum bestimmen
    mn, mx = diff.min(), diff.max()

    # Lineare Streckung auf den Bereich [0, 255]
    if mx == mn:
        return np.zeros_like(diff, dtype=np.uint8)

    lin_scaled_image = ((diff - mn) / (mx - mn) * 255.0).astype(np.uint8)
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
    candidates = []
    candidates.append(join(image_folder, "Testbild Rauschen 640x480.png"))
    candidates.append(join(image_folder, "Testbild_Rauschen_640x480.png"))

    image = None
    for p in candidates:
        if exists(p):
            image = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                break
    if image is None:
        print("Fehler beim Laden des Bildes. Bitte Pfad und Dateinamen prüfen.")
        return

    src = np.asarray(image, dtype=np.float32) / 255.0
    win = "Ex. 2) Rauschen"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Startwerte: eps0=1.0 -> 100, iters=500, lambda=0.1 -> 100
    cv2.createTrackbar("eps0 x100", win, 100, 2000, lambda v: None)
    cv2.createTrackbar("iters", win, 500, 1000, lambda v: None)
    cv2.createTrackbar("lambda x1000", win, 100, 1000, lambda v: None)

    # Initiale Berechnung und Anzeige
    e = cv2.getTrackbarPos("eps0 x100", win)
    it = cv2.getTrackbarPos("iters", win)
    l = cv2.getTrackbarPos("lambda x1000", win)
    eps0 = max(1, e) / 100.0
    iters = max(1, it)
    lambda_ = max(1, l) / 1000.0
    out = diffusion_filter(i=src, eps0=eps0, iters=iters, lambda_=lambda_)
    cv2.imshow(win, out)

    last_vals = (e, it, l)
    while True:
        e = cv2.getTrackbarPos("eps0 x100", win)
        it = cv2.getTrackbarPos("iters", win)
        l = cv2.getTrackbarPos("lambda x1000", win)
        if (e, it, l) != last_vals:
            eps0 = max(1, e) / 100.0
            iters = max(1, it)
            lambda_ = max(1, l) / 1000.0
            out = diffusion_filter(i=src, eps0=eps0, iters=iters, lambda_=lambda_)
            cv2.imshow(win, out)
            last_vals = (e, it, l)

        key = cv2.waitKey(50) & 0xFF
        if key in (27, ord('q')):
            cv2.imwrite("ex2_rauschen.png", out)
            break
        if key in (ord('s'),):
            cv2.imwrite("ex2_rauschen.png", out)

    cv2.destroyAllWindows()

def exercise2_run(image_folder=".", eps0: float = 1.0, iters: int = 500, lambda_: float = 0.1, filename: str = None):
    # Funktion nicht mehr verwendet; beibehalten, falls benötigt
    candidates = []
    if filename is not None:
        candidates.append(join(image_folder, filename))
    candidates.append(join(image_folder, "Testbild Rauschen 640x480.png"))
    candidates.append(join(image_folder, "Testbild_Rauschen_640x480.png"))

    image = None
    for p in candidates:
        if exists(p):
            image = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                break
    if image is None:
        print("Fehler beim Laden des Bildes. Bitte Pfad und Dateinamen prüfen.")
        return
    img_arr = np.asarray(image, dtype=np.float32) / 255.0
    out_img = diffusion_filter(i=img_arr, eps0=float(eps0), iters=int(iters), lambda_=float(lambda_))
    cv2.imshow("Ex. 2) Rauschen", out_img)
    cv2.waitKey(0)
    cv2.imwrite("ex2_rauschen.png", out_img)
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
    source_images = dirname(__file__)

    # ------------------
    # --- EXERCISE 1 ---
    # ------------------

    # exercise1(image_folder=source_images)

    # ------------------
    # --- EXERCISE 2 ---
    # ------------------

    exercise2(image_folder=source_images)

    # ------------------
    # --- EXERCISE 3 ---
    # ------------------

    # exercise3(image_folder=source_images)

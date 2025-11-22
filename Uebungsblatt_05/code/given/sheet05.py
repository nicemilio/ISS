import argparse

import cv2
import numpy as np
from os.path import *



def binarization(gray_u8, thresholds):
    if gray_u8 is None:
        return None
    img = gray_u8.astype(np.uint8)
    if isinstance(thresholds, (int, np.integer, np.uint8)):
        thr = int(thresholds)
        out = np.zeros_like(img, dtype=np.uint8)
        out[img > thr] = 255
        return out
    if thresholds is None:
        return img.copy()
    thr_list = list(np.asarray(thresholds).astype(int).ravel())
    thr_list = [min(255, max(0, t)) for t in thr_list]
    bins = np.array(sorted(thr_list)) if len(thr_list) > 0 else np.array([], dtype=int)
    labels = np.digitize(img, bins, right=False).astype(np.uint8)
    return labels


def exercise1(image_folder=".", input: str = None):
    try:
        image = cv2.imread(join(image_folder, input), cv2.IMREAD_GRAYSCALE)

    except FileNotFoundError:
        print("Fehler beim Laden des Bildes. Bitte Pfad und Dateinamen prüfen.")
        return
    if image is None:
        print("Fehler beim Laden des Bildes. Bitte Pfad und Dateinamen prüfen.")
        return
    cv2.namedWindow("Interaktive Binarisierung", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("n_thresholds", "Interaktive Binarisierung", 1, 9, lambda x: None)
    for i in range(1, 10):
        cv2.createTrackbar(f"T{i}", "Interaktive Binarisierung", i * 25 if i < 10 else 0, 255, lambda x: None)
    cv2.createTrackbar("auto_sort", "Interaktive Binarisierung", 1, 1, lambda x: None)
    palette = np.array([
        [0, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [128, 128, 128],
        [255, 128, 0],
    ], dtype=np.uint8)
    while True:
        n = cv2.getTrackbarPos("n_thresholds", "Interaktive Binarisierung")
        n = max(0, min(9, n))
        thrs = []
        for i in range(1, n + 1):
            thrs.append(cv2.getTrackbarPos(f"T{i}", "Interaktive Binarisierung"))
        if cv2.getTrackbarPos("auto_sort", "Interaktive Binarisierung") == 1 and len(thrs) > 0:
            thrs = sorted(thrs)
        if n <= 1 and len(thrs) >= 1:
            labels_or_binary = binarization(image, thresholds=int(thrs[0]))
            if labels_or_binary is None:
                break
            vis = cv2.cvtColor(labels_or_binary, cv2.COLOR_GRAY2BGR)
        else:
            labels = binarization(image, thresholds=thrs if len(thrs) > 0 else [])
            if labels is None:
                break
            k = int(labels.max()) if labels.size > 0 else 0
            k = min(k, 9)
            colors = palette[: k + 1]
            vis = colors[labels]
        cv2.imshow("Interaktive Binarisierung", vis)
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord('q')):
            break
        if key in (ord('s'),):
            cv2.imwrite("Ex. 1) Interaktive Binarisierung.png", vis)
    cv2.destroyAllWindows()


def connected_component_labeling(source: np.ndarray, neighborhood: int = 8):
    if source is None:
        return None
    img = source
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent = [0]
    def uf_find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def uf_union(a, b):
        ra, rb = uf_find(a), uf_find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb
    next_label = 1
    if neighborhood == 8:
        neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]
    else:
        neigh = [(-1, 0), (0, -1)]
    for y in range(h):
        for x in range(w):
            if img[y, x] == 0:
                neighbor_labels = []
                for dy, dx in neigh:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w:
                        lab = labels[yy, xx]
                        if lab != 0:
                            neighbor_labels.append(lab)
                if neighbor_labels:
                    m = int(min(neighbor_labels))
                    labels[y, x] = m
                    for lab in neighbor_labels:
                        while len(parent) <= max(m, lab):
                            parent.append(len(parent))
                        if parent[m] == 0:
                            parent[m] = m
                        if parent[lab] == 0:
                            parent[lab] = lab
                        uf_union(m, lab)
                else:
                    labels[y, x] = next_label
                    while len(parent) <= next_label:
                        parent.append(len(parent))
                    parent[next_label] = next_label
                    next_label += 1
    flat = {}
    cur = 1
    for y in range(h):
        for x in range(w):
            lab = labels[y, x]
            if lab != 0:
                root = uf_find(lab)
                if root not in flat:
                    flat[root] = cur
                    cur += 1
                labels[y, x] = flat[root]
    num_labels = cur - 1
    if num_labels <= 0:
        color = np.full((h, w, 3), 255, dtype=np.uint8)
        return color
    hsv = np.zeros((num_labels + 1, 1, 3), dtype=np.uint8)
    for k in range(1, num_labels + 1):
        hsv[k, 0, 0] = np.uint8((k * 179) // max(1, num_labels))
        hsv[k, 0, 1] = 255
        hsv[k, 0, 2] = 200
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    mask_fg = labels > 0
    out[mask_fg] = rgb[labels[mask_fg], 0]
    return out


def exercise3(image_folder=".", threshold: np.uint8=120):
    try:
        image = cv2.imread(join(image_folder, "Testbild_Werkzeuge_768x576.png"), cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError:
        print("Fehler beim Laden des Bildes. Bitte Pfad und Dateinamen prüfen.")
        return

    binarized_image = binarization(image, thresholds=threshold)
    labeled_image = connected_component_labeling(binarized_image)

    cv2.imshow("Ex. 3) Region Labeling", labeled_image)
    cv2.waitKey(0)
    saved_1 = cv2.imwrite("Ex. 3) Region Labeling.png", labeled_image)
    print("Gespeichert:", saved_1)
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Interaktive Binarisierung")
    parser.add_argument("--input", type=str, default="Polyeder_256x256.png")
    parser.add_argument("--threshold", type=np.uint8, default=120, help="Threshold für Aufgabe 3")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    image_folder = dirname(__file__)

    # ------------------
    # --- EXERCISE 1 ---
    # ------------------

    exercise1(image_folder=image_folder, input=args.input)


    # ------------------
    # --- EXERCISE 3 ---
    # ------------------

    exercise3(image_folder=image_folder, threshold=args.threshold)




import argparse
import numpy as np
import cv2
from os.path import *
import heapq
import math


# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def compute_gradient_magnitude(img):
    """Berechnet Gradientenbetrag (Sobel) + speichert Visualisierung."""
    Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(Gx**2 + Gy**2).astype(np.float64)

    vis = (255 * (magnitude / (magnitude.max() + 1e-9))).astype(np.uint8)
    cv2.imwrite("grad_magnitude.png", vis)

    return magnitude


def neighbors_8connected(r, c, rows, cols):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc, math.hypot(dr, dc)


def dijkstra_path(grad, start, end):
    """Dijkstra-Pfad zwischen Start und Endpunkt."""
    rows, cols = grad.shape
    sr, sc = start
    er, ec = end

    gmod = 0.5 * (grad[sr, sc] + grad[er, ec])

    dist = np.full((rows, cols), np.inf)
    prev = -np.ones((rows, cols, 2), dtype=np.int32)
    visited = np.zeros((rows, cols), dtype=bool)

    pq = []
    dist[sr, sc] = 0
    heapq.heappush(pq, (0, sr, sc))

    while pq:
        d, r, c = heapq.heappop(pq)
        if visited[r, c]:
            continue
        visited[r, c] = True

        if (r, c) == (er, ec):
            break

        for nr, nc, dd in neighbors_8connected(r, c, rows, cols):
            if visited[nr, nc]:
                continue

            cost = abs(grad[nr, nc] - gmod)
            nd = d + dd * cost + 1e-6

            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                prev[nr, nc] = (r, c)
                heapq.heappush(pq, (nd, nr, nc))

    # Backtracking
    if not np.isfinite(dist[er, ec]):
        return []

    path = []
    cur = (er, ec)
    seen = set()
    while True:
        path.append(cur)
        if cur == (sr, sc):
            break
        pr, pc = prev[cur]
        if pr < 0:
            break
        if (pr, pc) in seen:
            break
        seen.add((pr, pc))
        cur = (pr, pc)

    return path[::-1]


def draw_path(img, path, color=(0,0,255)):
    out = img.copy()
    for i in range(len(path)-1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        cv2.line(out, (c1, r1), (c2, r2), color, 1, cv2.LINE_AA)
    return out


def mask_from_paths(shape, paths):
    mask = np.zeros(shape, dtype=np.uint8)
    for path in paths:
        pts = np.array([[c, r] for r, c in path], dtype=np.int32).reshape(-1,1,2)
        cv2.polylines(mask, [pts], False, 255, 1)

    # Füllen der größten Kontur
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        areas = [cv2.contourArea(c) for c in cnts]
        i = int(np.argmax(areas))
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, cnts, i, 255, -1)
        return filled
    return mask


# ------------------------------------------------------------
# GUI – Mouse Callback
# ------------------------------------------------------------

click_points = []      # speichert gesetzte Punkte
paths = []             # speichert alle Pfade
grad_global = None     # Gradienten
img_color_global = None
img_display_global = None


def mouse_callback(event, x, y, flags, param):
    global click_points, paths
    global grad_global, img_color_global, img_display_global

    # Linksklick → Startpunkt
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((y, x))
        print(f"Startpunkt gesetzt: {(y, x)}")

    # Rechtsklick → Endpunkt → Pfad wird berechnet
    if event == cv2.EVENT_RBUTTONDOWN:
        if len(click_points) == 0:
            print("Bitte zuerst Startpunkt setzen (Linksklick).")
            return
        
        start = click_points[-1]
        end = (y, x)
        click_points.append(end)

        print(f"Berechne Pfad von {start} nach {end} ...")
        path = dijkstra_path(grad_global, start, end)
        if len(path) == 0:
            print("Kein Pfad gefunden.")
            return

        paths.append(path)
        img_display_global = draw_path(img_display_global, path, color=(0,255,0))


# ------------------------------------------------------------
# Hauptfunktion gemäß Aufgabenstellung
# ------------------------------------------------------------

def exercise1(image_folder=".", input: str = None):
    global grad_global, img_color_global, img_display_global

    image_path = join(image_folder, input)
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print("Fehler beim Laden des Bildes.")
        return

    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    grad = compute_gradient_magnitude(img_gray)
    grad_global = grad
    img_color_global = img_color
    img_display_global = img_color.copy()

    cv2.namedWindow("Intelligent Scissors", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Intelligent Scissors", mouse_callback)

    print("GUI gestartet.")
    print("Linksklick = Startpunkt")
    print("Rechtsklick = Endpunkt → Pfad")
    print("q = Speichern & Beenden")

    while True:
        cv2.imshow("Intelligent Scissors", img_display_global)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    # ---------------------
    # Speichern der Ergebnisse
    # ---------------------

    # erstes Segment extra speichern
    if paths:
        first = paths[0]
        cv2.imwrite("initial_path.png", draw_path(img_color, first))

    # Overlay aller Pfade
    overlay = img_color.copy()
    for p in paths:
        overlay = draw_path(overlay, p, color=(0,255,0))
    cv2.imwrite("is_overlay.png", overlay)

    # Maske
    mask = mask_from_paths(img_gray.shape, paths)
    cv2.imwrite("is_mask.png", mask)

    print("Gespeicherte Dateien:")
    print(" - grad_magnitude.png")
    print(" - initial_path.png")
    print(" - is_overlay.png")
    print(" - is_mask.png")
    print("Fertig.")




def exercise3(image_folder="."):
    try:
        image = cv2.imread(join(image_folder, "Testbild_Lena_512x512.png"), cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError:
        print("Fehler beim Laden des Bildes. Bitte Pfad und Dateinamen prüfen.")
        return

    # Todo Ihre Lösung


def parse_args():
    parser = argparse.ArgumentParser(description="Intelligent Scissors")
    parser.add_argument("--input", type=str, default="Testbild_Gangman_300x200.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    image_folder = "/Sheet06/code/given/"

    # ------------------
    # --- EXERCISE 1 ---
    # ------------------

    exercise1(image_folder=".", input=args.input)


    # ------------------
    # --- EXERCISE 3 ---
    # ------------------

    exercise3(image_folder=image_folder)




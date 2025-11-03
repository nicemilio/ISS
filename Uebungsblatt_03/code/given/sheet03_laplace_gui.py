import cv2
import sys

from io_utils import load_image_grayscale
from laplace import laplace_edges
from pathlib import Path

VARIANTS = ["L4", "L8"]

def on_change(_=None):
    pass # nothing to do here

def run_gui(img_path: str | Path):
    
    gray = load_image_grayscale(img_path).astype("float32") / 255.0

    cv2.namedWindow("Laplace Viewer", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("variant (0:L4, 1:L8)", "Laplace Viewer", 0, 1, on_change)
    cv2.createTrackbar("connectivity (0:4, 1:8)", "Laplace Viewer", 0, 1, on_change)
    cv2.createTrackbar("epsilon x1e-6", "Laplace Viewer", 1, 50, on_change)  # eps = slider * 1e-6

    while True: # get current mode and threshold, apply operator and update shown image

        v_idx = cv2.getTrackbarPos("variant (0:L4, 1:L8)", "Laplace Viewer")
        c_idx = cv2.getTrackbarPos("connectivity (0:4, 1:8)", "Laplace Viewer")
        eps_scale = cv2.getTrackbarPos("epsilon x1e-6", "Laplace Viewer")
        variant = VARIANTS[v_idx]
        eps = float(eps_scale) * 1e-6
        connectivity = 4 if c_idx == 0 else 8

        edges = laplace_edges(gray, variant=variant, connectivity=connectivity, eps=eps)

        
        vis = edges

        cv2.imshow("Laplace Viewer", vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord('q')):  # ESC or q
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_gui(sys.argv[1] if len(sys.argv) > 1 else "Lena_512x512.png")

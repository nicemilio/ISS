import cv2
import sys
import numpy as np

from pathlib import Path

from io_utils import load_image_grayscale
from sobel import sobel_gradients, magnitude_and_orientation, linear_stretch_to_u8
from viz import orientation_to_bin, render_orientation_bins, apply_magnitude_brightness, mask_by_threshold

MODES = ["magnitude", "orientation", "both"]

def on_change(_=None):
    pass # nothing to do here

def run_gui(img_path: str | Path):
    """
    Interactive Sobel visualization GUI.
    Shows magnitude, 4-bin orientation, or both combined.
    """

    # === Load grayscale image ===
    gray = load_image_grayscale(img_path).astype("float32")

    # === Compute Sobel gradients and derived quantities ===
    gx, gy = sobel_gradients(gray)
    magnitude, orientation = magnitude_and_orientation(gx, gy)
    magnitude_u8 = linear_stretch_to_u8(magnitude)

    # === Convert orientation from radians to degrees [0,180) ===
    theta_deg = (np.degrees(orientation) + 180.0) % 180.0
    orientation_bins = orientation_to_bin(theta_deg)
    orientation_rgb = render_orientation_bins(orientation_bins)

    # === Setup window and controls ===
    cv2.namedWindow("Sobel Viewer", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("mode (0:mag, 1:ori, 2:both)", "Sobel Viewer", 0, 2, on_change)
    cv2.createTrackbar("threshold", "Sobel Viewer", 0, 255, on_change)

    while True:
        mode_idx = cv2.getTrackbarPos("mode (0:mag, 1:ori, 2:both)", "Sobel Viewer")
        thr = cv2.getTrackbarPos("threshold", "Sobel Viewer")
        mode = MODES[mode_idx]

        # === Visualization modes ===
        if mode == "magnitude":
            vis = cv2.cvtColor(magnitude_u8, cv2.COLOR_GRAY2BGR)
            vis = mask_by_threshold(vis, magnitude_u8, thr)

        elif mode == "orientation":
            vis = mask_by_threshold(orientation_rgb, magnitude_u8, thr)

        elif mode == "both":
            vis = apply_magnitude_brightness(orientation_rgb, magnitude_u8)
            vis = mask_by_threshold(vis, magnitude_u8, thr)

        else:
            vis = cv2.cvtColor(magnitude_u8, cv2.COLOR_GRAY2BGR)

        cv2.imshow("Sobel Viewer", vis)

        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord("q")):  # ESC or q
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_gui(sys.argv[1] if len(sys.argv) > 1 else "Lena_512x512.png")

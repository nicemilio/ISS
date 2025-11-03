import cv2
import sys

from pathlib import Path

from io_utils import load_image_grayscale
from sobel import sobel_gradients, magnitude_and_orientation, linear_stretch_to_u8
from viz import orientation_to_bin, render_orientation_bins, apply_magnitude_brightness, mask_by_threshold

MODES = ["magnitude", "orientation", "both"]

def on_change(_=None):
    pass # nothing to do here

def run_gui(img_path: str | Path):
    
    # TODO: load data, apply operator with initial value

    cv2.namedWindow("Sobel Viewer", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("mode (0:mag, 1:ori, 2:both)", "Sobel Viewer", 0, 2, on_change)
    cv2.createTrackbar("threshold", "Sobel Viewer", 0, 255, on_change)

    while True: # get current mode and threshold, apply operator and update shown image
        
        mode_idx = cv2.getTrackbarPos("mode (0:mag, 1:ori, 2:both)", "Sobel Viewer")
        thr = cv2.getTrackbarPos("threshold", "Sobel Viewer")

        # TODO: Ihre Lösung
        vis = None

        cv2.imshow("Sobel Viewer", vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord('q')):  # ESC or q
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_gui(sys.argv[1] if len(sys.argv) > 1 else "Lena_512x512.png")

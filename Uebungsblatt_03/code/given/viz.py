import numpy as np
import cv2

# four bins centered at 0°, 45°, 90°, 135° with +-22.5° half-width
BIN_CENTERS = np.array([0.0, 45.0, 90.0, 135.0], dtype=np.float32)
HALF_WIDTH = 22.5

# fixed BGR colors per bin (distinct and legible)
BIN_COLORS = np.array([
    [  0,   0, 255],  # red
    [255,   0,   0],  # blue
    [  0, 255,   0],  # green
    [255, 255,   0],  # cyan
], dtype=np.uint8)

def orientation_to_bin(theta_deg: np.ndarray) -> np.ndarray:
    bin_idx = np.digitize(theta_deg, BIN_CENTERS - HALF_WIDTH) - 1
    bin_idx = bin_idx % len(BIN_CENTERS)
    return bin_idx

def render_orientation_bins(bin_idx: np.ndarray) -> np.ndarray:
    # Map bin indices to colors
    return BIN_COLORS[bin_idx]

def apply_magnitude_brightness(rgb: np.ndarray, mag_u8: np.ndarray) -> np.ndarray:
    # Normalize magnitude to [0, 1]
    mag_normalized = mag_u8.astype(np.float32) / 255.0
    # Expand dimensions for broadcasting
    mag_normalized = mag_normalized[:, :, np.newaxis]
    # Apply brightness scaling
    brightened_rgb = (rgb.astype(np.float32) * mag_normalized).astype(np.uint8)
    return brightened_rgb

def mask_by_threshold(rgb: np.ndarray, mag: np.ndarray, thr: float) -> np.ndarray:
    # Create a binary mask based on the magnitude threshold
    mask = (mag > thr).astype(np.uint8)
    # Apply the mask to each channel
    masked_rgb = rgb * mask[:, :, np.newaxis]
    return masked_rgb

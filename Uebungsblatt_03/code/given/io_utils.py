import cv2
import numpy as np

from pathlib import Path

def load_image_grayscale(path: str | Path) -> np.ndarray:

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img.astype(np.float32)

def save_u8(path: str | Path, img_u8: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img_u8)

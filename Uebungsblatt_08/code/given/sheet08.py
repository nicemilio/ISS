import numpy as np
import cv2
import argparse
import math 

def harris_corner_detector(image, kappa=0.15, sigma=2.0, threshold=500000):
    """
    Harris Corner Detector

    Parameters:
    - image: Input grayscale image (numpy array)
    - kappa: Harris detector parameter (typically 0.04-0.06)
    - sigma: Standard deviation for Gaussian smoothing
    - threshold: Threshold for corner response

    Returns:
    - corners: List of corner coordinates [(x, y), ...]

    """
    def calc_weight(u, v):
        e = np.exp(1)
        harris_weight = e ** ( - (u ** 2 + v ** 2) / (2 * (sigma ** 2)))
        return harris_weight

    # Arbeitskopie in float32, um Überläufe zu vermeiden
    img = image.astype(np.float32)

    dx = np.zeros(image.shape, dtype=np.float32)
    dy = np.zeros(image.shape, dtype=np.float32)

    for x in range(1, image.shape[0]-1):
        for y in range(1, image.shape[1]-1):
             dx[x,y] = (img[x+1,y] - img[x-1,y])
             dy[x,y] = (img[x,y+1] - img[x,y-1])
    
    dxx = np.zeros(dx.shape, dtype=np.float32)
    dyy = np.zeros(dy.shape, dtype=np.float32)

    for x in range(1, len(dx)): 
        for y in range(1, len(dx[x])): 
            dxx[x,y] = dx[x,y] ** 2
            dyy[x,y] = dy[x,y] ** 2

    dxy = np.zeros(dx.shape, dtype=np.float32)

    for x in range (1, len(dx)): 
        for y in range (1, len(dx[x])): 
            dxy[x,y] = dx[x,y] * dy[x,y]

    wdxx = np.zeros(dxx.shape, dtype=np.float32)
    wdyy = np.zeros(dyy.shape, dtype=np.float32)
    wdxy = np.zeros(dxy.shape, dtype=np.float32)

    # Feste Fenstergröße 3x3, Offsets u,v in {-1,0,1}
    H, W = dxx.shape
    for x in range(1, H-1): 
        for y in range(1, W-1): 
            sxx = 0.0
            syy = 0.0
            sxy = 0.0
            for u in (-1, 0, 1):
                for v in (-1, 0, 1):
                    w = calc_weight(u, v)
                    sxx += dxx[x+u, y+v] * w
                    syy += dyy[x+u, y+v] * w
                    sxy += dxy[x+u, y+v] * w
            wdxx[x,y] = sxx
            wdyy[x,y] = syy
            wdxy[x,y] = sxy

    response = np.zeros(wdxx.shape, dtype=np.float32)
    for x in range(1, H-1): 
        for y in range(1, W-1): 
            det = wdxx[x,y] * wdyy[x,y] - (wdxy[x,y] ** 2)
            trace = wdxx[x,y] + wdyy[x,y]
            response[x,y] = det - kappa * (trace ** 2)

    corners = []
    for x in range(1, H-1): 
        for y in range(1, W-1): 
            if response[x,y] > threshold: 
                corners.append((y,x))
    


    print(f"Harris Corner Detector - Parameter: κ={kappa}, σ={sigma}, t={threshold}")

    return corners


def visualize_corners(image, corners):
    """
    Visualize detected corners on image

    Args:
        image: Original grayscale image
        corners: List of corner coordinates

    Returns:
        result_image: RGB image with marked corners
    """
    # Konvertierung zu RGB falls Grauwert
    if len(image.shape) == 2:
        result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        result_image = image.copy()

    for corner in corners:
        x, y = corner
        y0 = max(0, y - 1)
        y1 = min(result_image.shape[0], y + 2)
        x0 = max(0, x - 1)
        x1 = min(result_image.shape[1], x + 2)
        result_image[y0:y1, x0:x1] = [255, 0, 0]

    return result_image


def compute_gradient_magnitude_orientation(image, keypoints, sigma=1.6):
    """
    Compute gradient magnitude and orientation around keypoints

    Args:
        image: Input grayscale image
        keypoints: List of keypoint coordinates [(x, y), ...]
        sigma: Standard deviation for Gaussian weighting

    Returns:
        magnitudes: Gradient magnitudes for each keypoint region
        orientations: Gradient orientations for each keypoint region
    """
    magnitudes = []
    orientations = []

    #TODO: Gradient Magnitude und Orientation berechnen
    print(f"Anzahl Keypoints: {len(keypoints)}, Sigma: {sigma}")

    return magnitudes, orientations


def compute_sift_descriptors(magnitudes, orientations, keypoints):
    """
    Compute SIFT descriptors from gradient information

    Args:
        magnitudes: Gradient magnitudes around keypoints
        orientations: Gradient orientations around keypoints
        keypoints: Keypoint coordinates

    Returns:
        descriptors: List of 128-dimensional SIFT descriptors (numpy arrays)
    """
    descriptors = []

    #TODO: SIFT Descriptors berechnen
    print(f"Anzahl Keypoints für Descriptors: {len(keypoints)}")

    return descriptors


def find_correspondences(descriptors1, descriptors2, keypoints1, keypoints2):
    """
    Find corresponding keypoints using minimum Euclidean distance

    Args:
        descriptors1: SIFT descriptors from image 1
        descriptors2: SIFT descriptors from image 2
        keypoints1: Keypoints from image 1
        keypoints2: Keypoints from image 2

    Returns:
        matches: List of matched keypoint pairs [(kp1_idx, kp2_idx, distance), ...]
    """
    matches = []

    #TODO: Korrespondenzfindung implementieren

    return matches


def visualize_correspondences(image1, image2, keypoints1, keypoints2, matches):
    """
    Visualize corresponding keypoints with same colors

    Args:
        image1, image2: Input images
        keypoints1, keypoints2: Keypoint coordinates
        matches: List of matched pairs

    Returns:
        result_image1, result_image2: Images with colored correspondences
    """
    # Konvertierung zu RGB
    if len(image1.shape) == 2:
        result_image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
    else:
        result_image1 = image1.copy()

    if len(image2.shape) == 2:
        result_image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)
    else:
        result_image2 = image2.copy()

    #TODO: Korrespondenz-Visualisierung implementieren

    print(f"Anzahl zu visualisierende Matches: {len(matches)}")

    return result_image1, result_image2


def load_image(image_path, grayscale=True):
    """Helper function to load images"""
    if grayscale:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")

    return image


def extract_corners_from_harris_result(harris_result_image):
    """
    Extrahiere Corner-Koordinaten aus Harris-Ergebnisbild (mit roten Markierungen)

    Args:
        harris_result_image: RGB-Bild mit rot markierten Corners

    Returns:
        corners: List of corner coordinates [(x, y), ...]
    """
    corners = []

    # Finde alle roten Pixel (255, 0, 0)
    red_mask = np.all(harris_result_image == [255, 0, 0], axis=2)

    if not np.any(red_mask):
        print("Keine roten Pixel gefunden!")
        return corners

    # Markiere bereits verarbeitete Pixel
    processed = np.zeros_like(red_mask, dtype=bool)

    height, width = red_mask.shape

    # Durchsuche das Bild nach 3x3 roten Bereichen
    for y in range(1, height - 1):  # Rand auslassen, da wir 3x3 prüfen
        for x in range(1, width - 1):

            # Prüfe ob Mittelpunkt rot ist und noch nicht verarbeitet
            if not red_mask[y, x] or processed[y, x]:
                continue

            # Prüfe ob kompletter 3x3 Bereich rot ist
            is_3x3_red = True
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if not red_mask[y + dy, x + dx]:
                        is_3x3_red = False
                        break
                if not is_3x3_red:
                    break

            # Wenn vollständiger 3x3 roter Bereich gefunden
            if is_3x3_red:
                # Mittelpunkt als Corner hinzufügen
                corners.append((x, y))

                # Alle 9 Pixel als verarbeitet markieren
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        processed[y + dy, x + dx] = True

    print(f"Aus Harris-Ergebnis extrahiert: {len(corners)} vollständige 3x3 Corner-Bereiche")
    return corners


def process_harris_task(image_path, kappa, sigma, threshold):
    """Aufgabe 2: Harris Corner Detector"""
    print("=== Aufgabe 2: Harris Corner Detector ===")
    print(f"Eingabebild: {image_path}")
    print(f"Parameter: κ={kappa}, σ={sigma}, t={threshold}")

    # Bild laden
    image = load_image(image_path, grayscale=True)
    print(f"Bildgröße: {image.shape}")

    # Harris Corner Detector
    corners = harris_corner_detector(image, kappa, sigma, threshold)

    print(f"Anzahl gefundener Corners: {len(corners)}")

    # Visualisierung
    result_image = visualize_corners(image, corners)

    # Ausgabe
    output_filename = f"result_harris_{image_path.split('/')[-1]}"
    cv2.imwrite(output_filename, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    print(f"Ergebnisbild gespeichert: {output_filename}")

    return corners, result_image


def process_sift_task(image1_path, image2_path, kappa, sigma, threshold, use_harris_results=False):
    """Aufgabe 4: Harris + SIFT + Matching"""
    print("=== Aufgabe 4: Harris + SIFT + Matching ===")
    print(f"Bild 1: {image1_path}")
    print(f"Bild 2: {image2_path}")
    print(f"Verwende Harris-Ergebnisse: {use_harris_results}")

    # Bilder laden
    if use_harris_results:
        # Lade Harris-Ergebnisbilder (RGB mit roten Punkten)
        harris_result1 = load_image(image1_path, grayscale=False)
        harris_result2 = load_image(image2_path, grayscale=False)

        # Extrahiere Corners aus roten Markierungen
        corners1 = extract_corners_from_harris_result(harris_result1)
        corners2 = extract_corners_from_harris_result(harris_result2)

        # Konvertiere zu Grauwert für SIFT-Berechnung
        image1 = cv2.cvtColor(harris_result1, cv2.COLOR_RGB2GRAY)
        image2 = cv2.cvtColor(harris_result2, cv2.COLOR_RGB2GRAY)

    else:
        # Normale Verarbeitung: Harris Corner Detection durchführen
        image1 = load_image(image1_path, grayscale=True)
        image2 = load_image(image2_path, grayscale=True)

        corners1 = harris_corner_detector(image1, kappa, sigma, threshold)
        corners2 = harris_corner_detector(image2, kappa, sigma, threshold)

    print(f"Corners Bild 1: {len(corners1)}, Corners Bild 2: {len(corners2)}")

    if len(corners1) == 0 or len(corners2) == 0:
        print("Keine Corners gefunden! Versuchen Sie andere Parameter.")
        return [], None, None

    # SIFT Descriptors berechnen
    print("\n--- SIFT Descriptor Berechnung ---")

    # Schritt 3: Gradient Magnitude und Orientation
    mags1, orients1 = compute_gradient_magnitude_orientation(image1, corners1, sigma)
    mags2, orients2 = compute_gradient_magnitude_orientation(image2, corners2, sigma)

    # Schritt 4: SIFT Descriptors
    desc1 = compute_sift_descriptors(mags1, orients1, corners1)
    desc2 = compute_sift_descriptors(mags2, orients2, corners2)

    print(f"SIFT Descriptors berechnet: {len(desc1)} für Bild 1, {len(desc2)} für Bild 2")

    # Korrespondenzfindung
    print("\n--- Korrespondenzfindung ---")
    matches = find_correspondences(desc1, desc2, corners1, corners2)

    print(f"Gefundene Korrespondenzen: {len(matches)}")

    # Visualisierung
    result1, result2 = visualize_correspondences(image1, image2, corners1, corners2, matches)

    # Ausgabe
    output1 = f"result_matching_{image1_path.split('/')[-1]}"
    output2 = f"result_matching_{image2_path.split('/')[-1]}"
    cv2.imwrite(output1, cv2.cvtColor(result1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(output2, cv2.cvtColor(result2, cv2.COLOR_RGB2BGR))
    print(f"Ergebnisbilder gespeichert: {output1}, {output2}")

    return matches, result1, result2


def main():
    parser = argparse.ArgumentParser(description='Harris Corner Detector und SIFT Descriptor')
    parser.add_argument('--task', choices=['harris', 'sift'], required=True,
                        help='Aufgabe: harris (Aufgabe 2) oder sift (Aufgabe 4)')
    parser.add_argument('--image1', required=True, help='Pfad zum ersten Bild')
    parser.add_argument('--image2', help='Pfad zum zweiten Bild (nur für SIFT)')
    parser.add_argument('--kappa', type=float, default=0.15, help='Harris Kappa Parameter')
    parser.add_argument('--sigma', type=float, default=2.0, help='Sigma für Gauss-Filter')
    parser.add_argument('--threshold', type=float, default=500000, help='Schwellwert für Corner Response')
    parser.add_argument('--use-harris-results', action='store_true',
                        help='Verwende Harris-Ergebnisbilder (mit roten Punkten) als Input für SIFT')

    args = parser.parse_args()

    try:
        if args.task == 'harris':
            process_harris_task(args.image1, args.kappa, args.sigma, args.threshold)

        elif args.task == 'sift':
            if not args.image2:
                raise ValueError("Für SIFT task wird --image2 benötigt")

            process_sift_task(args.image1, args.image2, args.kappa, args.sigma, args.threshold, args.use_harris_results)

    except Exception as e:
        print(f"Fehler: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\nProgramm erfolgreich beendet!")
    return 0


if __name__ == "__main__":
    exit(main())
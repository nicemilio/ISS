import cv2
import numpy as np

def checkerboard():

    image = np.zeros((8, 8, 3), dtype=np.uint8)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):

            if (x + y) % 2 == 0:
                # OpenCV arbeitet mit einer BGR-Konvention
                # Der erste Wert entspricht dem Blau-Anteil, usw.
                image[y, x] = (0, 0, 255) 

            # Zeilen referenzieren y-Koordinaten
            # Spalten referenzieren x-Koordinaten
            pixel_value = image[y, x]
            print(f"(x={x}, y={y}): {pixel_value}")

    # Vergroessere das Bild zum Anzeigen
    zoom_factor = 50
    zoomed_image = cv2.resize(image, (image.shape[1] * zoom_factor, image.shape[0] * zoom_factor), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Checkerboard", zoomed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    checkerboard()

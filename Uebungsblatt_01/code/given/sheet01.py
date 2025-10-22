import cv2
import numpy as np

from os.path import *

def exercise2a(a: int, k: int, image_folder="."):

    image = cv2.imread(join(image_folder, "Lena_512x512.png"))

    # TODO: Ihre Lösung

    cv2.imshow("Ex. 2a)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def exercise2b(a: int, k: int, image_folder="."):

    image = cv2.imread(join(image_folder, "Lena_512x512.png"))

    # TODO: Ihre Lösung
    
    cv2.imshow("Ex. 2b)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def exercise3(image_folder="."):
    
    image = cv2.imread(join(image_folder, "Lena_512x512.png"))
    mask = cv2.imread(join(image_folder, "Maske_Lena_512x512.png"))

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):

            if not mask[y, x].any():
                image[y, x] = (0, 0, 0) 

    cv2.imshow("Ex. 3)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # ------------------
    # --- EXERCISE 2 ---
    # ------------------

    a = 5 # scale factor
    k = 2 # channel

    exercise2a(a=a, k=k)
    exercise2b(a=a, k=k)

    # ------------------
    # --- EXERCISE 3 ---
    # ------------------

    exercise3()

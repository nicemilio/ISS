import argparse

import cv2
from os.path import *


def exercise1(image_folder=".", input: str = None):
    image_path = join(image_folder, input)
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError:
        print("Fehler beim Laden des Bildes. Bitte Pfad und Dateinamen prüfen.")
        return

    # Todo Ihre Lösung
    #  Achten sie darauf, den Dijkstra Algorithmus wie in der Vorlesung beschrieben (Folien 38ff.) zu implementieren.
    



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

    exercise1(image_folder=image_folder, input=args.input)


    # ------------------
    # --- EXERCISE 3 ---
    # ------------------

    exercise3(image_folder=image_folder)




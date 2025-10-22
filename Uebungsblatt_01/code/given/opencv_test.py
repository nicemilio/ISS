import cv2
import numpy as np

def iss_test():

    image = np.zeros((300, 300, 3), dtype=np.uint8)

    color = (255, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    thickness = 5

    text = "ISS"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2

    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)

    cv2.imshow("Test image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    iss_test()

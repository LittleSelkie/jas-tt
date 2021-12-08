import cv2
from cv_helper import Image

if __name__ == "__main__":
    image = Image('landscape1.jpg')
    image.save('test.jpg')
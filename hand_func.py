"""
That module use mediapipe hands to detect hands and landmarks.
It allows to crop hand from photos and reshapes it to square.
ALso gets landmarsks to extract features.
"""

# Package

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image

# Function

class handDetector():
    """
    CLass that based on media pipe hands. It used preyrained model to find hand and its landmarks.
    """
    def __init__(self,mode=True,maxHands=1,modelComplexity=1,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.modelComplex,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img):

        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)


        return img

    def findPosition(self, img, handNo=0):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]

            for id,lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                lmlist.append([id,cx,cy])

        return lmlist


def get_points(detector, img):
    """
    It returns landmarks of shown hand
    :param detector: mediapipe hands class
    :param img: nparray
    :return: list of lists
    """

    img_with_hands = detector.findHands(img)
    rect_coordinates = detector.findPosition(img)

    return rect_coordinates

def get_rectangle_id(detector, img):
    """
    It returns points that will cropp an image to a rectangle around the hand
    :param detector: mediapipe hand class
    :param img: nparray    :return:
    """


    img_with_hands = detector.findHands(img)
    results = detector.findPosition(img)


    try:

        min_x = min(results, key=lambda x: x[1])[1]
        max_x = max(results, key=lambda x: x[1])[1]
        min_y = min(results, key=lambda x: x[2])[2]
        max_y = max(results, key=lambda x: x[2])[2]

        image_height, image_width, _ = img.shape

        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        bbox_size = max(bbox_width, bbox_height)

        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        start_x = max(center_x - bbox_size // 2)
        start_y = max(center_y - bbox_size // 2, 0)
        end_x = min(center_x + bbox_size // 2, image_width)
        end_y = min(center_y + bbox_size // 2, image_height)

        cropped_image = img[start_y:end_y, start_x:end_x]


        return cropped_image

    except:

        return None




def show_result(img, extra_space, detector):
    """
    It shows rectangle on images based on mediapipe hands model.
    :param img: nparray
    :param extra_space: int
    :param detector: mediapipe hand class
    :return: plot
    """

    img_with_hands = detector.findHands(img)
    img_with_position = detector.findPosition(img)

    results = img_with_position

    min_1 = min(results, key=lambda x: x[1])[1]
    max_1 = max(results, key=lambda x: x[1])[1]
    min_2 = min(results, key=lambda x: x[2])[2]
    max_2 = max(results, key=lambda x: x[2])[2]

    cv2.rectangle(img_with_hands, (min_1 - extra_space, max_2 + extra_space), (max_1 + extra_space, min_2 - extra_space), (0, 0, 255), 2)
    cv2.imshow('Hands Detected', img_with_hands)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cut_image_to_rectangle(img, detector):
    """
    It cut an image to rectangle based on mediapipe hands pretrained model
    :param img: nparray
    :param detector: mediapipe hands class
    :return: nparray
    """
    extra_space = 20

    img_with_hands = detector.findHands(img)
    img_with_position = detector.findPosition(img)

    results = img_with_position

    if len(results) != 0:
        min_1 = min(results, key=lambda x: x[1])[1]
        max_1 = max(results, key=lambda x: x[1])[1]
        min_2 = min(results, key=lambda x: x[2])[2]
        max_2 = max(results, key=lambda x: x[2])[2]

        cropped_img = img_with_hands[min_2 - extra_space:max_2 + extra_space, min_1 - extra_space:max_1 + extra_space]

        return cropped_img
    else:
        return None


def create_square(image):
    """
    It resizes an image to get square
    :param image: nparray   :return:
    """

    try:
        x, y, _ = image.shape

        new_shape = max(x, y)

        image = tf.image.resize_with_crop_or_pad(image, new_shape, new_shape)

        return image.numpy()

    except:


        return None

def small_square(img):
    """
    It resizes an image to 100x100 size
    :param img: nparray   :return:
    """

    pil_img = Image.fromarray(img)
    pil_img.thumbnail((100, 100))  # thumbnail modifies the image in place and returns None

    return np.array(pil_img)

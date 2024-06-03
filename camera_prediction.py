"""

This module is created to detect numbers from hands in-real time

"""

# Packages

import cv2
from hand_func import handDetector, get_points
import numpy as np
from joblib import dump, load
from collections import Counter
import math
import pandas as pd
import time
from sklearn.preprocessing import Normalizer
import pygame
import torch
import ast
from count_fingers_cnn import predict_result
from hand_func import cut_image_to_rectangle, create_square, small_square

# It will be shown based on pygame

pygame.init()
pygame.font.init()
font = pygame.font.SysFont('Arial', 30)

# Functions

def get_points(detector, img):
    """
    :param detector: Mediapipe Hands class
    :param img: nparray
    :return: rectangle's coordinates (list)

    """

    img_with_hands = detector.findHands(img)
    rect_coordinates = detector.findPosition(img)

    return rect_coordinates


def calculate_vector(point1, point2):
    """
    That function is to calculate vectores based on two points
    :param point1: list1 -> [idx, x, y] (representation from mediapipe hands)
    :param point2: list2 -> [idx, x, y] (representation from mediapipe hands)
    :return: vector (nparray)
    """
    return np.array([point2[1] - point1[1], point2[2] - point1[2]])

def split_lists(list_lists):
    """
    That function splits one list of lists to several smaller one. It is based on mediapipe hands.
    It converts it to lists of every finger.

    :param list_lists: list of lists [[idx, x, y] ...]
    :return: thumb, idx_finger, mid_finger, ring_finger, pinky_finger (lists of lists)
    """

    thumb = list_lists[1:5]
    idx_finger = list_lists[5:9]
    mid_finger = list_lists[9:13]
    ring_finger = list_lists[13:17]
    pinky_finger = list_lists[17:21]

    return thumb, idx_finger, mid_finger, ring_finger, pinky_finger

def get_vector_len(data, idx, point1, point2):
    """
    It calculates vector from calculate_vector function and then calculates its length
    :param data: list of lists (for example thumb list)
    :param idx: int (thumb is 0, index finger is 1, middle finger is 2, ring finger is 3, pinky finger is 4)
    :param point1: list1 -> [idx, x, y] (representation from mediapipe hands)
    :param point2: list2 -> [idx, x, y] (representation from mediapipe hands)
    :return: vector, length (list, float)
    """

    vector = calculate_vector(data[idx][point1], data[idx][point2])
    length = math.sqrt(vector[0] ** 2 + vector[1] ** 2)

    return vector, length

def calculate_angle(vector1, vector2, length_v1, length_v2):

    """
    That function calculate angles between chosen fingers

    :param vector1: list    :param vector2:
    :param length_v1: float    :param length_v2:
    :return: theta
    """

    dot_product1 = np.dot(vector1, vector2)
    cos_theta1 = dot_product1 / (length_v1 * length_v2)
    cos_theta1 = np.clip(cos_theta1, -1.0, 1.0)
    theta1 = np.degrees(np.arccos(cos_theta1))

    return theta1

def get_finger_angles(list_lists):

    """
    It calculates angles between all fingers. You have to just give list of landmarks from mediapipe hands.
    :param list_lists: list of lists    :return:  list (mean angle o every space between fingers)
    """

    mean_angles = []
    thumb, idx_finger, mid_finger, ring_finger, pinky_finger = split_lists(list_lists)
    fingers = [thumb, idx_finger, mid_finger, ring_finger, pinky_finger]
    for idx in range(len(fingers) - 1):

        vector11, length_v11 = get_vector_len(fingers, idx, 0, 2)
        vector12, length_v12 = get_vector_len(fingers, idx, 1, 3)

        vector21, length_v21 = get_vector_len(fingers, idx + 1, 0, 2)
        vector22, length_v22 = get_vector_len(fingers, idx + 1, 1, 3)

        if length_v11 != 0 and length_v21 != 0:

            theta1 = calculate_angle(vector11, vector21, length_v11, length_v21)

        else:
            theta1 = 0

        if length_v12 != 0 and length_v22 != 0:

            theta2 = calculate_angle(vector12, vector22, length_v12, length_v22)

        else:
            theta2 = 0

        theta = np.mean([theta1, theta2])
        mean_angles.append(theta)

    return mean_angles

def calculate_mean_angles(points):
    """

    :param points: list of lists    :return:  of mean angles
    """
    return get_finger_angles(points)

def finger_length(list_):
    """
    It calculates length of fingers
    :param list_: list (landmarks from one finger)
    :return: float (length)
    """
    return np.sqrt((list_[0][1] - list_[3][1]) ** 2 + (list_[0][2] - list_[3][2]) ** 2)

def distance_between(list1, list2):
    """
    It calculates distance between fingers
    :param list1: list     :param list2:
    :return: float (length)
    """

    return np.sqrt((list1[1] - list2[1])**2 + (list1[2] - list2[2])**2)

def count_differences(list_lists):
    """
    That function automatize couting distances between fingers
    :param list_lists: list of lists ([[idx, x, y]...]
    :return: list
    """

    distances = []

    thumb, idx_finger, mid_finger, ring_finger, pinky_finger = split_lists(list_lists)
    fingers = [thumb, idx_finger, mid_finger, ring_finger, pinky_finger]

    for idx in range(len(fingers) - 1):
        distance = (distance_between(fingers[idx][-1], fingers[idx + 1][-1])
                    / (np.mean([finger_length(fingers[idx]), finger_length(fingers[idx + 1])])))
        distances.append(distance)

    return distances

def check_finger_place(list_lists):
    """
    It compares place of finger's tip and choosen places. It showed if finger is crossed.
    :param list_lists: list of lists
    :return: list
    """

    distances = []

    thumb, idx_finger, mid_finger, ring_finger, pinky_finger = split_lists(list_lists)
    fingers = [thumb, idx_finger, mid_finger, ring_finger, pinky_finger]

    thumb_place = (distance_between(fingers[0][-1], list_lists[0]) / (np.mean(finger_length(fingers[0]))))

    distances.append(thumb_place)

    for idx in range(1, len(fingers)):

        finger_place = (distance_between(fingers[idx][-1], list_lists[17]) / (np.mean(finger_length(fingers[idx]))))

        distances.append(finger_place)

    return distances

def split_distances(distances):
    """
    Splitting distances to more columns
    :param distances: list of lists
    :return: pandas Series
    """
    return pd.Series({
        'first_distance': distances[0],
        'second_distance': distances[1],
        'third_distance': distances[2],
        'fourth_distance': distances[3]
    })

def split_place(distances):
    """
    Splitting places to more columns
    :param distances: list of lists
    :return: pandas Series
    """
    return pd.Series({
        'first_place': distances[0],
        'second_place': distances[1],
        'third_place': distances[2],
        'fourth_place': distances[3]
    })
def split_angles(angles):
    """
    Splitting angles to more columns
    :param angles: list of lists
    :return: pandas Series
    """
    return pd.Series({
        'first_angle': angles[0],
        'second_angle': angles[1],
        'third_angle': angles[2],
        'fourth_angle': angles[3]
    })

def normalize_columns(df, columns):
    """
    Normalizing data on chosen columns, by norm L2.
    From a practical standpoint, L1 tends to shrink coefficients to zero whereas L2 tends to shrink coefficients evenly.
    L1 is therefore useful for feature selection, as we can drop any variables associated with coefficients that
    go to zero. L2, on the other hand, is useful when you have collinear/codependent features.
    :param df: pandas DataFrame
    :param columns: list
    :return: pandas DataFrame
    """
    normalizer = Normalizer(norm='l2')
    normalized_values = normalizer.fit_transform(df[columns])
    return pd.DataFrame(normalized_values, columns=columns)

ANGLE_COLUMNS = ['first_angle', 'second_angle', 'third_angle', 'fourth_angle']
DISTANCE_COLUMNS = ['first_distance', 'second_distance', 'third_distance', 'fourth_distance']
PLACE_COLUMNS = ['first_place', 'second_place', 'third_place', 'fourth_place']

def capture_frames(cap):
    """
    Get frames from camera and collect five frames
    :param cap: cv2.VideoCapture class
    :return: list of lists
    """
    frames = []
    while len(frames) < 3:
        ret, frame = cap.read()
        if not ret:
            print("Nie udało się pobrać klatki")
            break
        frames.append(frame)
    return frames

def process_frames(df):
    """
    Preprocessing frames to get features: angles, distances and places
    :param df: pandas DataFrame   :return:
    """

    df['fingers_angles'] = df['points'].apply(calculate_mean_angles)
    angles_split = df['fingers_angles'].apply(split_angles)
    df = pd.concat([df, angles_split], axis=1)

    df['fingers_distances'] = df['points'].apply(count_differences)
    distances_split = df['fingers_distances'].apply(split_distances)
    df = pd.concat([df, distances_split], axis=1)

    df['fingers_place'] = df['points'].apply(check_finger_place)
    distances_split = df['fingers_place'].apply(split_place)
    df = pd.concat([df, distances_split], axis=1)

    return df


def process_frame_cnn(df, filter):
    """
    Preprocess images to get square size 100x100, filtered and in tensor.
    :param df: pandas DataFrame
    :param filter: function
    :return: pytorch tensor
    """

    df['image'] = df['rectangle'].apply(create_square)
    df['image'] = df['image'].apply(small_square)
    df['image'] = df['image'].apply(filter)

    X = np.array(df['image'].tolist())

    X = torch.tensor(X, dtype=torch.float)

    return X

def count_result_cnn(X_tensor, model):
    """
    Calculate the most common prediction based on CNN chosen model.
    :param X_tensor: pytorch tensor
    :param model: pytorch class nn.module
    :return: int
    """

    results = []

    for X in X_tensor:
        X = X.unsqueeze(0)
        X = X.unsqueeze(0)

        result = predict_result(model, X)


        results.append(result.squeeze())

    most_common_prediction = Counter(results).most_common(1)[0][0]

    return most_common_prediction

def count_result(data, model):

    """
    Calculate the most common prediction based on SVM chosen model.
    :param data: pandas DataFrame
    :param model: sklearn model
    :return: predictions, labels, time2 - time1 (lists, lists, float)
    """

    data['fingers_angles'] = data['points'].apply(calculate_mean_angles)
    angles_split = data['fingers_angles'].apply(split_angles)
    data = pd.concat([data, angles_split], axis=1)

    data['fingers_distances'] = data['points'].apply(count_differences)
    distances_split = data['fingers_distances'].apply(split_distances)
    data = pd.concat([data, distances_split], axis=1)

    data['fingers_place'] = data['points'].apply(check_finger_place)
    distances_split = data['fingers_place'].apply(split_place)
    data = pd.concat([data, distances_split], axis=1)

    fingers_data = data[['first_angle', 'second_angle', 'third_angle', 'fourth_angle', 'first_distance', 'second_distance',
                         'third_distance', 'fourth_distance', 'first_place', 'second_place', 'third_place',
                         'fourth_place', 'class']]

    angle_columns = ['first_angle', 'second_angle', 'third_angle', 'fourth_angle']
    normalized_angles = normalize_columns(fingers_data, angle_columns)

    distance_columns = ['first_distance', 'second_distance', 'third_distance', 'fourth_distance']
    normalized_distances = normalize_columns(fingers_data, distance_columns)

    place_columns = ['first_place', 'second_place', 'third_place', 'fourth_place']
    normalized_place = normalize_columns(fingers_data, place_columns)

    fingers_data[angle_columns] = normalized_angles
    fingers_data[distance_columns] = normalized_distances
    fingers_data[place_columns] = normalized_place

    time1 = time.time()
    fingers_data = fingers_data.dropna()
    time2 = time.time()

    features = fingers_data.drop(columns=['class'])
    labels = fingers_data['class']

    predictions = model.predict(features)

    return predictions, labels, time2 - time1


def predict_frames(df, model):
    """
    Preprocess data to prepared representation. Giving most common prediction
    :param df: pandas DataFrame
    :param model: sklearn class
    :return: list, int
    """

    normalized_angles = normalize_columns(df, ANGLE_COLUMNS)
    normalized_distances = normalize_columns(df, DISTANCE_COLUMNS)
    normalized_place = normalize_columns(df, PLACE_COLUMNS)

    df[ANGLE_COLUMNS] = normalized_angles
    df[DISTANCE_COLUMNS] = normalized_distances
    df[PLACE_COLUMNS] = normalized_place
    new_data = df[ANGLE_COLUMNS + DISTANCE_COLUMNS + PLACE_COLUMNS]

    predictions = model.predict(new_data)

    most_common_prediction = Counter(predictions).most_common(1)[0][0]

    return predictions, most_common_prediction

def get_camera_data(model_path, svm, model = None, filter = None):
    """
    That function connects us to camera and save frames. Based on that we got prediction of counting the fingers.
    :param model_path: string
    :param svm: sklearn class model
    :param model: pytorch class nn.model
    :param filter: function
    :return: pygame plot
    """

    cap = cv2.VideoCapture(0)
    detector = handDetector()
    if svm:
        best_model = load(model_path)
    else:
        #best_model = model()
        #best_model.load_state_dict(torch.load(f"{model_path}.pt"))
        best_model = load(model_path)

    if not cap.isOpened():
        print("Nie można otworzyć kamery")
        exit()

    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption('Hand Gesture Recognition')

    try:
        while True:
            frames = capture_frames(cap)

            df = pd.DataFrame({'image': frames})

            if svm:

                df['points'] = df['image'].apply(lambda img: get_points(detector, img))

                try:
                    df = process_frames(df)
                    predictions, most_common_prediction = predict_frames(df, best_model)
                except IndexError:

                    most_common_prediction = None
            else:

                df['image'] = df['image'].apply(lambda img: cut_image_to_rectangle(img, detector))

                try:
                    X_tensor = process_frame_cnn(frames, detector, filter)
                    most_common_prediction = count_result_cnn(X_tensor, best_model)
                except AttributeError:

                    most_common_prediction = None

            frame = frames[0]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(frame_rgb.transpose(1, 0, 2))

            screen.blit(frame_surface, (0, 0))

            if most_common_prediction is not None:
                text_surface = font.render(f'Prediction: {most_common_prediction}', True, (255, 0, 0))
                screen.blit(text_surface, (10, 10))

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    finally:
        cap.release()
        cv2.destroyAllWindows()

# get_camera_data('SVC()-model.joblib', svm=True)


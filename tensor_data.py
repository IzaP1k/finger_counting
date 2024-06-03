# Packages

import os
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch
import cv2
from hand_func import cut_image_to_rectangle, create_square, small_square
from count_fingers_cnn import predict_result
import time

"""
That module consists of functions that allows us to create tensor data
"""

# Functions
def grayscale(image_path):

    with Image.open(image_path) as img:

        img = np.array(img)

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(gray_image.shape)

        return gray_image

def get_data_folder(path=r"boxesdata"):

    FOLDER_PATH = path

    NAME_PATH = FOLDER_PATH + r"\labels"
    PHOTO_PATH = FOLDER_PATH + r"\images"

    FILENAME_LIST = []
    CONTENT_LIST1 = []
    CONTENT_LIST2 = []
    CONTENT_LIST3 = []
    CONTENT_LIST4 = []
    IMAGE_LIST = []

    for filename in os.listdir(NAME_PATH):
        FILE_PATH = os.path.join(NAME_PATH, filename)
        with open(FILE_PATH, 'r', encoding='utf-8') as file:
            content = file.read()
            content = content.replace("0 ", "")
            content = content.replace("\n", " ")
            content_list = content.split()
            CONTENT_LIST1.append(content_list[0])
            CONTENT_LIST2.append(content_list[1])
            CONTENT_LIST3.append(content_list[2])
            CONTENT_LIST4.append(content_list[3])

        FILENAME_LIST.append(filename)

    for image_name in os.listdir(PHOTO_PATH):
        image_path = os.path.join(PHOTO_PATH, image_name)
        normalized_img = grayscale(image_path)
        IMAGE_LIST.append(normalized_img)


    data = {
        'filename': FILENAME_LIST,
        'image': IMAGE_LIST,
        'x1': CONTENT_LIST1,
        'x2': CONTENT_LIST2,
        'x3': CONTENT_LIST3,
        'x4': CONTENT_LIST4
    }
    df = pd.DataFrame(data)

    return df

def split_tensor_data(data_boxes):

    X = np.stack(data_boxes['image'].values)
    y = data_boxes[['x1', 'x2', 'x3', 'x4']].astype(float).values

    print(X.shape)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=123)

    X_train = torch.tensor(X_train, dtype=torch.float)
    X_val = torch.tensor(X_val, dtype=torch.float)

    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=20)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=20)

    return train_loader, val_loader


def change_img(img_path):

    img = cv2.imread(img_path)
    img = np.array(img)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    tensor_image = torch.tensor(gray_image, dtype=torch.float32)

    tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)

    return tensor_image

def count_tensor_data(data_boxes, image, label):

    X = np.stack(data_boxes[image].values)
    y = data_boxes[label].astype(float).values


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=123)

    X_train = torch.tensor(X_train, dtype=torch.float)
    X_val = torch.tensor(X_val, dtype=torch.float)

    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=50)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=50)

    return train_loader, val_loader


def prepare_image(data, filter):


    data['image'] = data['rectangle'].apply(create_square)

    time1 = time.time()
    data = data.dropna(subset=['image'])
    time2 = time.time()

    data['image'] = data['image'].apply(small_square)
    data['image'] = data['image'].apply(filter)

    X = np.array(data['image'].tolist())

    X = torch.tensor(X, dtype=torch.float)

    return X, time2 - time1, data

def get_answer(test_tensor, model):
    results = []

    for X in test_tensor:
        X = X.unsqueeze(0)
        X = X.unsqueeze(0)

        results.append(predict_result(model, X))

    return np.mean(results)
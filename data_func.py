import numpy as np
import os
import cv2
from skimage import exposure
from skimage.feature import hog
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

'''

Funkcje do czyszczenia i przekształcania danych

'''

def clean_Nan_value(data):

    data['points'] = data['points'].apply(lambda x: np.nan if x == [] else x)

    data = data.dropna(subset=['rectangle'])

    data['rectangle'] = data['rectangle'].apply(lambda x: np.nan if len(x) == 0 else x)

    data = data.dropna(subset=['rectangle'])

    data = data.dropna(subset=['points'])

    return data


def get_grayscale(image):

    try:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image_gray

    except:

        return None


def hog_apply(image):

    try:

        fd, hog_image = hog(image,
                            orientations=8,
                            pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1),
                            visualize=True,
                            channel_axis=-1, )

        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        return hog_image_rescaled
    except:
        return None

def laplace_opencv(image):

    try:
        image = cv2.GaussianBlur(image, (7, 7), 0)  # drugi argument zmniejszone z 3, 3
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered_image = cv2.Laplacian(image_gray, cv2.CV_16S, ksize=7)  # ksize zwiększone z 3

        return filtered_image
    except:
        return None

def show_filtered_data(filtered, col, numb):

    for i in range(numb):
        image = filtered.loc[i, col]
        label = filtered.loc[i, 'class']
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'Obraz {i+1}, o fladze {label}')
        plt.show()



def one_hot_encoding(df):

    one_hot_encoded_df = pd.get_dummies(df['class'], prefix='class')

    df = pd.concat([df, one_hot_encoded_df], axis=1)

    return df
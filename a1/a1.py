import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf


def data(img_path, label_path, task):
    '''
    img_path: directorty of images
    label_path: string, path of labels

    Reads the label dataframe, loads images from directory to array and divides it into train / valid / test sets
    '''
    df = pd.read_csv(label_path, sep='\s+')
    df.replace(to_replace=-1, value=0, inplace=True)  # replace -1 by 0

    if task == '1'
        y = df.gender
    if task == '2'
        y = df.smiling

    X = []
    for x in df.img_name:
        X.append(cv2.imread(img_path + x))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=df.smiling)

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    return np.array(X_train), y_train, np.array(X_test), y_test

def data_test(img_path, label_path,task):
    '''
    img_path: directorty of images
    label_path: string, path of labels

    Reads the label dataframe, loads images from directory to array and divides it into train / valid / test sets
    '''
    df = pd.read_csv(label_path, sep='\s+')
    df.replace(to_replace=-1, value=0, inplace=True)  # replace -1 by 0

    if task == '1'
        y = df.gender
    if task == '2'
        y = df.smiling

    X = []
    for x in df.img_name:
        X.append(cv2.imread(img_path + x))

    y_test = tf.keras.utils.to_categorical(y)

    return np.array(X), y_test
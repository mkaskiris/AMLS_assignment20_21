import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def data_preprocessing_test(path, task):
    DIR, img_dir, labels = get_path(path)
    df = load_data_cartoon(DIR)
    if task == 'eyes':
        X,y = get_eyes(img_dir,df)
    if task == 'faces':
        X,y = get_faces(img_dir,df)
    return X,y

def data_preprocessing(path, task):
    DIR, img_dir, labels = get_path(path)
    df = load_data_cartoon(DIR)
    if task == 'eyes':
        X,y = get_eyes(img_dir,df)
    if task == 'faces':
        X,y = get_faces(img_dir,df)
    return split(X,y)

def get_path(path):
    DIR = path
    img_folder = path + 'img/'
    labels = path + 'labels.csv'

    return DIR, img_folder, labels

def crop_eyes(img_path): #Crops one eye
    img = cv2.imread(img_path)
    return cv2.resize(img[250:280,190:220],(10,10))

def crop_face(img_path): #Crops lower part of face / jawline
    img = cv2.imread(img_path)
    return cv2.resize(img[280:400,150:350],(50,32))


def get_faces(img_dir, df): #appends all faces into a list with their respective labels
    X = []
    y = []

    for x in df.iterrows():
        X.append(crop_faces(img_dir + x[0]))
        y.append(x[1][1])

    return np.array(X).reshape(len(X), 4800), np.array(y)


def get_eyes(img_dir, df):  # removes glasses and returns smaller images of eyes
    comp = [[255, 255, 255]]  # white row

    X = []
    y = []

    for i in df.iterrows():
        ori = crop_eyes(img_dir + i[0])  # read and crop the eye
        flat = np.array(ori).reshape(100, 3)
        for j in range(len(flat) - 1):  # compare rows
            if (flat[j] == comp[0]).all():  # if the row is completelly white
                X.append(ori)
                y.append(i[1][0])
                break
    return np.array(X).reshape(len(X), 300), np.array(y)


def load_data_cartoon(DIR): #load data
    df = pd.read_csv(DIR + 'labels.csv', sep='\s+')
    df.set_index('file_name', inplace=True)

    return df


def split(X, y): #train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42,
     #                                                 stratify=y_train)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    #X_val = np.array(X_val)
    #y_val = np.array(y_val)

    return X_train, y_train, X_test, y_test #X_val, y_val,


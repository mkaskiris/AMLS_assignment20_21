from imutils import face_utils
import dlib
import imutils
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
def get_data(DIR):
    '''
    DIR: string, directory where contraining image folder and labels.csv

    passes data to get_landmarks and receives landmarks . 
    splits it into train / test / valid
    '''
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')

    img_folder = DIR + 'img/'
    label = DIR + 'labels.csv'
    df = pd.read_csv(label, sep = '\s+')

    features = []
    smile = []

    for name in df.img_name:
        x = get_landmarks(DIR + name)
        if x is not None:
            features.append(x)
            smile.append(df.loc[df.img_name == name].smiling)

    X = np.array(features).reshape(len(features), 68 * 2).astype(float)
    y = np.array(smile).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test


def get_landmarks(img_path):
    '''
    img_path: directory of images 

    gets 68 landmarks from images
    '''
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    shape = None
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

    return shape

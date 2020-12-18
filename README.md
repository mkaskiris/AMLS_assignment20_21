# AMLS_assignment20_21
1. Put dataset in dataset folder
2. For task A dataset folder should  be called 'celeba'
3. For task B dataset folder should be called 'cartoon_set'
4. Each folder should contain another folder named 'img' containing the images and a 'labels.csv' containing the labels
5. the labels.csv file should be of the same form as with the ones released with the assignment

Note: need to put shape_predictor_68_face_landmarks.dat in folder a2 for the facial landmarks extraction to work. File is too big for github

Ive put another model (Transfer learning, uncomment it if you cannot provide the .dat file for the landmarks

Ignore submission and util folders 

REQUIREMENTS: 
General: Numpy, Pandas, Matplotlib, OpenCV
Metrics from sklearn: StratifiedKFold, GridSearchCV, train_test_split, metrics, pipeline, classification_report, confusion_matrix
Feature extraction: imutils, dlib
Transfer Learning: Tensorflow, Keras, tf-nightly
Models: Inception_V3, xgb, SVM, KNN, LDA

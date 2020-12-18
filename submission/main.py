from a1.a1 import data
from b1.b1 import data_preprocessing
from a2.a2 import get_data
import tensorflow as tf
import keras
import numpy as np
import padas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import xgboost as xgb

# Task A1
X_train,y_train,X_test,y_test,X_val,y_val = data('./dataset/celeba/img/', './dataset/celeba/labels.csv')
model_A1 = keras.models.load_model('./a1/gender_model') #load pre train model
history_fine = gender_model.fit(X_train,y_train,epochs=1,validation_data=(X_val,y_val))   #train 1 epoch
acc_A1_train = history_fine.history['accuracy']         
loss, acc_A1_test = model_A1.evaluate(X_test,y_test)   # Test model based on the test set.


# ======================================================================================================================
# Task A2
X_train,y_train,X_test,y_test = get_data('./dataset/celeba/')
model_A2 = xgb.XGBClassifier(learning_rate=0.3,n_estimators=70, max_depth=5,min_child_weight=1,gamma=0,colsample_bytree=0.3)
acc_A2_train = model_A2.fit(X_train,y_train)
acc_A2_test = model_A2.evaluate(X_test,y_test)


# ======================================================================================================================
# Task B1
X_train,y_train,X_test,y_test = data_preprocessing('./dataset/cartoon_set/', 'faces')
model_B1 = LinearDiscriminantAnalysis()
acc_B1_train = model_B1.train(X_train,y_train)
acc_B1_test = model_B1.score(X_test,y_test)


# ======================================================================================================================
# Task B2
X_train,y_train,X_test,y_test = data_preprocessing('./dataset/cartoon_set/', 'eyes')
model_B2 = LinearDiscriminantAnalysis()
acc_B2_train = model_B2.train(X_train,y_train)
acc_B2_test = model_B2.score(X_test,y_test)

# ======================================================================================================================
## Print out your results with following format:
print('TA1:{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(   acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))
import warnings
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet, Lasso,RidgeClassifier
import tensorflow.keras as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,f1_score,accuracy_score,roc_auc_score,roc_curve

from sklearn.linear_model import LogisticRegressionCV, ElasticNet, ElasticNetCV,LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler


import mlflow
from urllib.parse import urlparse
from mlflow.models.signature import infer_signature
import mlflow.sklearn
# import mlflow.tensorflow
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

'''
Part one:  Use mlflow to automate experimentation of model reduction. 
Track values for Hyperparameter tunning of alpha and gamma values for ElasticNet model

Part 2: Use mlflow to automate experimentation tracking for Linear and nonlinear 
dimensionality reductions 

PCA (Linear dimensionality reduction): values of principal components and predict it with 
Logistic regression to track the optimal number of dimensions to reduce features to.

Autoencoders (Non-linear dimensionality reduction): # of dimensions to reduce our model to

'''

# Read Data

x= pd.read_csv("../../../data/X_train.csv") 
y = pd.read_csv("../../../data/Y_train.csv")
scaler = StandardScaler()
x = scaler.fit_transform(x)
y.replace({'C':0,'AD':1}, inplace = True)
# x_test = pd.read_csv("../data/X_test.csv")
y_trial = pd.read_csv("../../../data/Y_train.csv")
y_trial.replace({'C':0,'AD':1}, inplace = True)
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.25, shuffle= True, random_state=1)

n_latent = float(sys.argv[1]) if len(sys.argv) > 1 else x_train.shape[1]
lr = float(sys.argv[2]) if len(sys.argv) > 1 else 1e-5
n_hidden = float(sys.argv[3]) if len(sys.argv) > 1 else x_train.shape[1]
activ = str(sys.argv[4]) if len(sys.argv) > 1 else "selu"



with mlflow.start_run():

    n_input = x_train.shape[1] 
    # n_hidden = x_train.shape[1]
    # n_latent = 57
    epoch = 500
    # lr = 1e-3
    encoder = Sequential([
        Dense(n_input, activation=activ, input_dim=n_input),
        Dense(n_hidden, activation=activ),
        Dense(n_latent, activation=activ)
        ])
    decoder = Sequential([
        Dense(n_hidden, activation=activ),
        Dense(n_input, activation=activ)
        ])
    stacked_ae = Sequential([encoder,decoder])
    checkpoint_cb = tf.callbacks.ModelCheckpoint("my__model.h5",save_best_only=True)
    early_stopping_cb = tf.callbacks.EarlyStopping(patience=3,restore_best_weights=True)
    stacked_ae.compile(optimizer = tf.optimizers.Adam(learning_rate = lr), loss = tf.losses.mean_absolute_error, metrics=['accuracy'])
    history = stacked_ae.fit(x_train, x_train, epochs=epoch, validation_data=[x_valid, x_valid],callbacks=[checkpoint_cb, early_stopping_cb], verbose=0)
    # history = stacked_ae.fit(x_train, x_train, epochs=epoch, validation_data=[x_valid, x_valid])
    train_encoded = encoder.predict(x_train)
    full_encoded = stacked_ae.predict(x_train)
    test_encoded = encoder.predict(x_valid)
    model_logit_ae = LogisticRegression(solver='liblinear', random_state=0).fit(train_encoded,y_train)
    f1_vals = f1_score(model_logit_ae.predict(test_encoded),y_valid)
    accur = accuracy_score(model_logit_ae.predict(test_encoded),y_valid)


    print("F1_score (n_latent={:f},learning_rate={:f},n_hidden={:f}, activation={})):".format(n_latent,lr,n_hidden,activ))
    print(classification_report(model_logit_ae.predict(test_encoded),y_valid))
    print("F1_score : %s" % f1_vals)
    print("Accuracy : %s" % accur)

    mlflow.log_param("n_latent",n_latent)
    mlflow.log_param("learning_rate",lr)
    mlflow.log_param("n_hidden",n_hidden)
    mlflow.log_param("activation",activ)

    mlflow.log_metric("f1_score", f1_vals)
    mlflow.log_metric("Accuracy", accur)

    mlflow.sklearn.log_model(stacked_ae,"model")
    # mlflow.sklearn.save_model('model',"../ENmodel_.pkl")
    # mlflow.log_artifact("elm.png")

    # predictions = elm.predict(x_train)
    # signature = infer_signature(x_train,predictions)

    # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


    # if tracking_url_type_store != "file": 
    #     mlflow.sklearn.log_model(
    #         elm,"model", registered_model_name="ElasticNetmodel", signature=signature
    #     )
    # else:
    #     mlflow.sklearn.log_model(elm,"model",signature=signature)


import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet, Lasso,RidgeClassifier
# import tensorflow.keras as tf
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,f1_score,accuracy_score,roc_auc_score,roc_curve
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

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
Part 2: Use mlflow to automate experimentation tracking for Linear and nonlinear 
dimensionality reductions 

PCA (Linear dimensionality reduction): values of principal components and predict it with 
Logistic regression to track the optimal number of dimensions to reduce features to optimum principal components.


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

n_dim = float(sys.argv[1]) if len(sys.argv) > 1 else 300
# l1_ratio = float(sys.argv[2]) if len(sys.argv) > 1 else 0.5

with mlflow.start_run():

    # dim = np.arange(30,len(x_train)+1, 10)
    # accur = []
    n_dim = np.int0(n_dim)
    pca = PCA(n_components=np.int0(n_dim))
    x_pca_train = pca.fit_transform(x_train)
    x_pca_valid = pca.transform(x_valid)
    model_logit = LogisticRegression(solver='liblinear', random_state=0).fit(x_pca_train,y_train)
    accur = accuracy_score(model_logit.predict(x_pca_valid),y_valid)
    # accur.append(accuracy_score(model_logit.predict(x_pca_valid),y_valid))
    # accur.append(accuracy_score(model_logit.predict(x_pca_valid),y_valid)) 
    print("principal_components={:d}:".format(n_dim))
    print("Accuracy : %s" % accur)


    mlflow.log_param("num_dimensions",n_dim)
    # mlflow.log_param("l1_ratio",l1_ratio)
    # mlflow.log_metric("f1_score", f1_vals)
    mlflow.log_metric("Accuracy", accur)

    mlflow.sklearn.log_model(model_logit,"model")
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


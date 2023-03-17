# do dimensionality reduction using PCA,autoencoder and then apply for models \
# applying reduced dimensions for (logit, svm and trees).
# compare penalized methods (logisticregressionCV, randomforestCV, XGBoostCV)
# Stacking models and graph neural networks
# Stack all complex methods as base learners and use ridge lasso and elasticnet as meta learer
###############################################################################
# 8 candidate models,all using dim. reduction (pca and ae) and lasso for feature reductnion  
from sklearn.linear_model import LogisticRegressionCV, ElasticNet, ElasticNetCV,Ridge,LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import pandas as pd
import random
import numpy as np
import tensorflow.keras as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,f1_score,accuracy_score,roc_auc_score,roc_curve
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
class dim_red:
    def pca_reduce(self,x_train, x_valid):
        pca = PCA(n_components=57)
        x_pca_train = pca.fit_transform(x_train)
        x_pca_valid = pca.transform(x_valid)
        x_pca_train = pd.DataFrame(x_pca_train)
        x_pca_valid = pd.DataFrame(x_pca_valid)
        x_pca_train['train_status'] = 'train' 
        x_pca_valid['train_status'] = 'test'
        df_pca = pd.concat([x_pca_train, x_pca_valid], axis = 0)
        return df_pca
    def ae_reduce(self, x_train, x_valid):
        n_input = len(x_train.columns)
        # n_hidden = len(x_train.columns)
        n_latent = 66
        epoch = 200
        lr = 1e-3
        encoder = Sequential([
            Dense(n_input, activation='selu', input_dim=n_input),
            # Dense(n_hidden, activation='selu'),
            Dense(n_latent, activation='selu')
        ])
        decoder = Sequential([
            # Dense(n_hidden, activation='selu'),
            Dense(n_input, activation='selu')
        ])
        stacked_ae = Sequential([encoder,decoder])
        checkpoint_cb = tf.callbacks.ModelCheckpoint("my__model.h5",save_best_only=True)
        early_stopping_cb = tf.callbacks.EarlyStopping(patience=3,restore_best_weights=True)
        stacked_ae.compile(optimizer = tf.optimizers.Adam(learning_rate = lr), loss = tf.losses.mean_squared_error, metrics='accuracy')
        # history = stacked_ae.fit(x_train, x_train, epochs=epoch, validation_data=[X_test, X_test],callbacks=[checkpoint_cb, early_stopping_cb])
        history = stacked_ae.fit(x_train, x_train, epochs=epoch, validation_data=[x_valid, x_valid], verbose = 0)
        train_encoded = encoder.predict(x_train)
        full_encoded = stacked_ae.predict(x_train)
        test_encoded = encoder.predict(x_valid)
        # print('mse:',mean_squared_error(stacked_ae.predict(full_encoded), x_train))
        train_encoded = pd.DataFrame(train_encoded)
        test_encoded = pd.DataFrame(test_encoded)
        train_encoded['train_status'] = 'train' 
        test_encoded['train_status'] = 'test'
        df_ae = pd.concat([train_encoded, test_encoded], axis = 0)
        return df_ae

class models:
    def models_(self,df,model, y_train, y_valid):
        train_encoded = df.loc[df['train_status'] =='train']
        train_encoded.drop(columns = ['train_status'], inplace = True)
        test_encoded = df.loc[df['train_status'] =='test']
        test_encoded.drop(columns = ['train_status'], inplace = True)
        if (model == 'log_reg'):
            model_ = LogisticRegression(solver='liblinear', random_state=0).fit(train_encoded,y_train)
        elif (model == 'svc'):
            model_ = svm.SVC()
        elif(model == 'xgb'):
            model_ = XGBClassifier(n_estimators=100,max_depth=2, learning_rate=0.05, alpha=1)
        elif(model == 'rfc'):
            model_ = RandomForestClassifier(max_depth=6,max_leaf_nodes=9)
        elif(model == 'lda'):
            model_ = LinearDiscriminantAnalysis()
        elif(model == 'elastic'):
            model_ = ElasticNet(alpha=0.1, l1_ratio=0.00)
        model_.fit(train_encoded, y_train)
        if (model == 'elastic'):
            yhat_train = model_.predict(train_encoded)
            yhat = model_.predict(test_encoded)
            yhat = pd.DataFrame(yhat)
            yhat = round(abs(yhat))
            yhat_train = pd.DataFrame(yhat_train)
            yhat_train = round(abs(yhat_train))
            print('Train classificationreport:',classification_report(yhat_train,y_train))
            print('Test classification_report:',classification_report(yhat,y_valid))
        else:
            print('Train classificationreport:',classification_report(model_.predict(train_encoded),y_train))
            print('Test classification_report:',classification_report(model_.predict(test_encoded),y_valid))
                


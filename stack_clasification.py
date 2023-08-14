# Project Mineria de Datos
# Universidad Nacionalde Ingenieria
# Maestria en Ciencias de la Computación- Facultad de Ciencias
# Programming by : Jorge Lozano
# Version : 1.00

#import imageio
from skimage import feature
import pandas as pd
#import os
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score, RandomizedSearchCV
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import joblib
from xgboost import XGBRegressor
import xgboost as xgb
from scipy.stats import zscore
import numpy as np
import seaborn as sns
import math
import copy
from scipy import stats
import librosa, librosa.display
from scipy import stats
from scipy.stats import kurtosis, skew


class Stacking_Clasification(object):
    def __init__(self, data,data_test, target,SEED,NFOLDS):
        self.data=data
        self.data_test=data_test
        self.target=target
        self.SEED= SEED
        self.NFOLDS=NFOLDS
    def Stacking_Models(self,knn,rf,et,gb,svc,gbm):
        X = self.data.drop(self.target, axis=1) # a0,a1,a2.....a9
        y = self.data[self.target].ravel()  # TYPE {0,1}
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=self.SEED)# Generame 30% Data Test
        #Concatenate dataframes with new data
        X_test=pd.concat([X_test,self.data_test], axis=0)
        X_test=X_test.reset_index(drop= True)
        y_test = np.append(y_test, 0)
        # Transform data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        ntrain = X_train.shape[0]
        ntest = X_test.shape[0]
        # Generate diferentes samples
        kf = KFold(n_splits= self.NFOLDS, shuffle=True, random_state=self.SEED)
        # Training Diferentes Models
        knn_oof_train, knn_oof_test = self.get_oof(knn, ntrain , ntest, self.NFOLDS,kf,X_train, y_train , X_test)
        rf_oof_train, rf_oof_test = self.get_oof(rf, ntrain , ntest, self.NFOLDS,kf,X_train, y_train , X_test)
        et_oof_train, et_oof_test = self.get_oof(et, ntrain , ntest, self.NFOLDS,kf,X_train, y_train , X_test)
        gb_oof_train, gb_oof_test = self.get_oof(gb, ntrain , ntest, self.NFOLDS,kf,X_train, y_train , X_test)
        svc_oof_train, svc_oof_test = self.get_oof(svc, ntrain, ntest, self.NFOLDS, kf, X_train, y_train, X_test)
        # Stacking results of each Models
        x_train = np.concatenate((rf_oof_train,et_oof_train,gb_oof_train,knn_oof_train,svc_oof_train), axis=1)
        x_test = np.concatenate((rf_oof_test,et_oof_test,gb_oof_test,knn_oof_test,svc_oof_test), axis=1)
        # Training Model with last results
        gbm.train(x_train,y_train)
        y_pred = gbm.predict(x_test)
        #Model Evaluation
        self.Models_Evaluation(y_train, knn_oof_train,rf_oof_train,et_oof_train,gb_oof_train,svc_oof_train)
        self.Models_General_Evaluation(y_test, y_pred)
        self.ML_Graphics(y_test, y_pred)
        return (y_pred[y_pred.size-1:]) # la prediccion el ultimo elemento del nparray

    def Models_Evaluation(self,y_train, knn_oof_train,rf_oof_train,et_oof_train,gb_oof_train,svc_oof_train):
        # Evaluación de modelos
        mcknn=confusion_matrix(y_train, knn_oof_train)
        mcrf=confusion_matrix(y_train, rf_oof_train)
        mcet=confusion_matrix(y_train, et_oof_train)
        mcgb=confusion_matrix(y_train, gb_oof_train)
        mcsv=confusion_matrix(y_train, svc_oof_train)
        print("KNN Model- Results Model")
        print("KNN-CM: {}".format(mcknn))
        print("Precision {}".format(mcknn[0,0]/(mcknn[0,0]+mcknn[0,1])))
        print("Recall {}".format(mcknn[0, 0] / (mcknn[0, 0] + mcknn[1, 0])))
        print("Bondad {}".format(mcknn[0, 0]*0.3 / (mcknn[0, 0] + mcknn[1, 0])+mcknn[0,0]*0.7/(mcknn[0,0]+mcknn[0,1])))
        print("Random Forest Model- Results Model")
        print("Rf-CM: {}".format(mcrf))
        print("Precision {}".format(mcrf[0, 0] / (mcrf[0, 0] + mcrf[0, 1])))
        print("Recall {}".format(mcrf[0, 0] / (mcrf[0, 0] + mcrf[1, 0])))
        print("Bondad {}".format(mcrf[0, 0] * 0.3 / (mcrf[0, 0] + mcrf[1, 0]) + mcrf[0, 0] * 0.7 / (mcrf[0, 0] + mcrf[0, 1])))
        print("ExtraTreesClassifier Model- Results Model")
        print("ET-CM: {}".format(mcet))
        print("Precision {}".format(mcet[0, 0] / (mcet[0, 0] + mcet[0, 1])))
        print("Recall {}".format(mcet[0, 0] / (mcet[0, 0] + mcet[1, 0])))
        print("Bondad {}".format(mcet[0, 0] * 0.3 / (mcet[0, 0] + mcet[1, 0]) + mcet[0, 0] * 0.7 / (mcet[0, 0] + mcet[0, 1])))
        print("GradientBoostingClassifier Model- Results Model")
        print("GB-CM: {}".format(mcgb))
        print("Precision {}".format(mcgb[0, 0] / (mcgb[0, 0] + mcgb[0, 1])))
        print("Recall {}".format(mcgb[0, 0] / (mcgb[0, 0] + mcgb[1, 0])))
        print("Bondad {}".format(mcgb[0, 0] * 0.3 / (mcgb[0, 0] + mcgb[1, 0]) + mcgb[0, 0] * 0.7 / (mcgb[0, 0] + mcgb[0, 1])))
        print("SVC Model- Results Model")
        print("SVC-CM: {}".format(mcsv))
        print("Precision {}".format(mcsv[0, 0] / (mcsv[0, 0] + mcsv[0, 1])))
        print("Recall {}".format(mcsv[0, 0] / (mcsv[0, 0] + mcsv[1, 0])))
        print("Bondad {}".format(mcsv[0, 0] * 0.3 / (mcsv[0, 0] + mcsv[1, 0]) + mcsv[0, 0] * 0.7 / (mcsv[0, 0] + mcsv[0, 1])))

    def Models_General_Evaluation(self,y_test, y_pred):
        mcsm = confusion_matrix(y_test, y_pred)
        print("Stacking Model- Results Model")
        print("STACKING MODELS CM: {}".format(mcsm))
        print("Precision {}".format(mcsm[0, 0] / (mcsm[0, 0] + mcsm[0, 1])))
        print("Recall {}".format(mcsm[0, 0] / (mcsm[0, 0] + mcsm[1, 0])))
        print("Bondad {}".format(mcsm[0, 0] * 0.3 / (mcsm[0, 0] + mcsm[1, 0]) + mcsm[0, 0] * 0.7 / (mcsm[0, 0] + mcsm[0, 1])))
        print("Data Test Descriptions")
        print(pd.DataFrame(y_test).describe())
        print("Data Prediction Descriptions")
        print(pd.DataFrame(y_pred).describe())

    def ML_Graphics(self,yt, yp):
        sns.jointplot(yt, yp, kind='kde', color="red")
        plt.show()
        sns.kdeplot(yt, label="Real")
        sns.kdeplot(yp, label="Prediction")
        plt.show()
        plt.plot(range(len(yt)), yt, color="Blue", label="Real")
        plt.plot(range(len(yp)), yp, color="Red", label="Prediction")
        plt.show()

    #Function to training models
    def get_oof(self, clf , ntrain , ntest, NFOLDS,kf,X_train, y_train , X_test):
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))
        for i, (train_index, test_index) in enumerate(kf.split(X_train)):
            x_tr = X_train[train_index] # Data X of training process
            y_tr = y_train[train_index] # Data y of training process
            x_te = X_train[test_index]  # Data x of test process
            clf.train(x_tr, y_tr)  # Training Data
            oof_train[test_index] = clf.predict(x_te) # Value of prediction with internal test data
            oof_test_skf[i, :] = clf.predict(X_test) # For each model testint the data external test data
        oof_test[:] = oof_test_skf.mean(axis=0) # Average of all splits generates
        return (oof_train.reshape(-1, 1), oof_test.reshape(-1, 1))  #models= [a1,a2,a3,a4,,,] where ai is an model

class SklearnWrapper(object):
    def __init__(self, clf, params=None):
        self.clf = clf(**params)
    def train(self, x_train, y_train):
        self.clf.fit(x_train,y_train)
    def predict(self, x):
        return self.clf.predict(x)

class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)
    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=np.log(y_train))
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)
    def predict(self, x):
        return np.exp(self.gbdt.predict(xgb.DMatrix(x)))

class GBM_Classifier(object):
    def __init__(self,clf,params=None):
        self.clf=clf(**params)
    def train(self,x_train,y_train):
        self.clf.fit(x_train,y_train)
    def predict(self,x):
        return self.clf.predict(x)


xgb_params={
    'learning_rate':0.95,
    'n_estimators':1000,
    'max_depth':4,
    'min_child_weight':2,
    'gamma':1,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'objective':'binary:logistic',
    'nthread':-1,
    'scale_pos_weight':1,
}

knn_params = {
        'n_neighbors': 11,
        'weights': 'distance',
        'algorithm': 'auto',
        'p': 1,
    }
rf_params = {
        'n_jobs': -1,
        'n_estimators': 300,
        # 'max_features': 'sqrt',
        'max_depth': 20,
        #warm_start': True,
        'random_state': 42,
        'min_samples_leaf': 2,
        'verbose': 0
    }
et_params = {
        'n_jobs': -1,
        'n_estimators': 600,
        # 'max_features': 0.5,
        'max_depth': 20,
        'min_samples_leaf': 2,
        'random_state': 42,
        #warm_start': True,
        'verbose': 0
    }
gb_params = {
        'n_estimators': 400,
        'max_depth': 5,
        'min_samples_leaf': 3,
        'verbose': 0,
    }
svc_params = {
        'kernel': 'linear',
        'C': 0.025,
    }

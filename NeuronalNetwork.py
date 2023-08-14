

import imageio
from skimage import feature
import pandas as pd
import os
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
class RedNeuronal(object):
    def __init__(self, data,target,SEED):
        self.data=data
        self.target= target
        self.SEED= SEED

    def Models_score_cv(self,clf, X, y):
        scores = cross_val_score(clf, X, y, cv=5, scoring='neg_mean_absolute_percentage_error')
        clf.fit(X, y)
        print("Score: {:.10f}".format((1 + np.mean(scores)) * 100))
        plt.clf()
        plt.plot(range(len(y)), clf.predict(X), "y-")
        plt.bar(range(len(y)), y)
        plt.show()
        return (1 + np.mean(scores)) * 100

    def Models_score_base(self,clf, X, y):
        clf.fit(X,y)
        score = clf.score(X, y)
        return score

    def Models_Evaluation(self,clf,X_train, X_test, y_train, y_test):
        # Evaluación de modelos
        clf.fit(X_train,y_train)
        y_pred= clf.predict(X_test)
        mcknn=confusion_matrix(y_test, y_pred)
        print("Results Model")
        print("Confusion Matrix: {}".format(mcknn))
        print("Report of Clasification Indicator")
        print(classification_report(y_test,y_pred))
        bnd= accuracy_score(y_test,y_pred)
        return bnd
    def Generate_Neuronal_Model_KNFold(self,activation,neu1,neu2,N_SP):
        # This code serves to fit model with activation function, neurons for each level
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target].ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=self.SEED)
        # Transform data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        n_train= X_train.shape[0]
        n_test= X_test.shape[0]
        # Create Model with 2 capas
        hidden_layer_sizes = (neu1, neu2)
        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=10000, random_state=42, learning_rate_init=0.1)
        ft=self.K_Fold(clf,N_SP,X_train,y_train,X_test, n_train , n_test) # This objetc contained all inforamtion about models generated in Kfolds
        best=-99
        k=0
        c_destino= 'models'
        savedir= os.path.join(os.getcwd(),c_destino)
        file_scaler= 'scaler.pkl'
        file_scaler = os.path.join(savedir, file_scaler)
        joblib.dump(scaler,file_scaler)
        model_return=[]
        for i ,clf_i in enumerate(ft):
          print("Start Results Model  :", i , " Fold")
          clft= joblib.load(clf_i) # This model was trained
          y_pred=clft.predict(X_test)
          mcknn=confusion_matrix(y_test, y_pred)
          print("Confusion Matrix: {}".format(mcknn))
          print("Report of Clasification Indicator")
          print(classification_report(y_test, y_pred))
          print("End Results Model  :", i , " Fold")
          ac=accuracy_score(y_test,y_pred)
          if ac> best:
            best=ac
            k=i
        best_model= joblib.load(ft[k])
        model_return.append(best_model)
        model_return.append(scaler)
        model_return.append(k)
        return model_return

    def Generate_Neuronal_Model(self,activation,neu1,neu2):
        # This code serves to fit model with activation function, neurons for each level
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target].ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=self.SEED)
        # Transform data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        # Create Model with 2 capas
        hidden_layer_sizes = (neu1, neu2)
        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=10000, random_state=42, learning_rate_init=0.1)
        clf.fit(X_train,y_train)
        mcknn=confusion_matrix(y_test, clf.predict(X_test))
        print("Results Model")
        print("Confusion Matrix: {}".format(mcknn))
        print("Report of Clasification Indicator")
        print(classification_report(y_test,clf.predict(X_test)))
        return clf,scaler
    def Generate_Prediction(self,clf,scaler,data_test):
        x_test = scaler.transform(data_test)
        y_pred= clf.predict(x_test)
        return y_pred
    def K_Fold(self,clf,N_SP,X_train,y_train,X_test,n_train,n_test):
        f_train = np.zeros((n_train,))
        f_test = np.zeros((n_test,))
        f_test_skf = np.empty((N_SP, n_test))
        kf= KFold(n_splits=N_SP,shuffle= True)
        kf.get_n_splits(X_train)
        c_destino= 'models'
        savedir= os.path.join(os.getcwd(),c_destino)
        models_name=[]
        for i, (train_index, test_index) in enumerate(kf.split(X_train)):
            x_tr = X_train[train_index] # Data X of training process
            y_tr = y_train[train_index] # Data y of training process
            x_te = X_train[test_index] # Data X of training process
            filename= 'model'+ str(i)+'.pkl'
            filename = os.path.join(savedir, filename)
            models_name.append(filename)
            clf.fit(x_tr,y_tr)
            joblib.dump(clf,filename)
            #f_train[test_index] = clf.predict(x_te) # Value of prediction with internal test data
            #f_test_skf[i, :] = clf.predict(X_test) # For each model testint the data external test data
        #f_test[:] = f_test_skf.mode(axis=0) # Mode of all splits generates}
        return models_name

    def GridSearch_Hyperparamters(self,n_cap1, n_cap2):
        # Cuadricula de posibles párametros
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target].ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=self.SEED)
        # Transform data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        layers=[]
        activation=['logistic', 'relu', 'identity', 'tanh']
        solver= ["lbfgs"]
        for i in range(1,n_cap1):
            for j in range(1,n_cap2):
                layers.append((i,j))
        params={"hidden_layer_sizes":layers,"solver":solver,"activation":activation}
        clf= MLPClassifier(max_iter=10000, random_state=42, learning_rate_init=0.1)
        clf= GridSearchCV(clf,params,cv=5,n_jobs=-1) # Usa Cross Validation
        clf.fit(X_train,y_train)
        return clf.best_params_

    def Indetify_Hyperparamters_Analysis(self,n_cap1, n_cap2):
        # This code serve to identify how many hidden_layers_sizes needs for each level and testing what activation function we needs
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target].ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=self.SEED)
        # Transform data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        best_score = -1
        best_activation = ""
        best_hidden_layer = []

        for activation in ['logistic', 'relu', 'identity', 'tanh']:
            for capa_1 in range(1, n_cap1):
                for capa_2 in range(1, n_cap2):
                    hidden_layer_sizes = (capa_1, capa_2)
                    clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=10000, random_state=42, learning_rate_init=0.1)
                    print("With {}".format(activation))
                    print("Layer sizes: {}".format(hidden_layer_sizes))
                    cur_score= self.Models_score_base(clf, X_train, y_train)
                    #cur_score = self.Models_Evaluation(clf,X_train, X_test, y_train, y_test)
                    if cur_score > best_score:
                        best_score = cur_score
                        best_activation = activation
                        best_hidden_layer = hidden_layer_sizes
        print("The Best Parameters is:")
        print(best_activation)
        print(best_hidden_layer)
        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=best_hidden_layer, activation=best_activation, max_iter=10000, random_state=42, learning_rate_init=0.1)
        cur_score = self.Models_Evaluation(clf,X_train, X_test, y_train, y_test)
        return best_activation,best_hidden_layer


import os
import sys
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models


# Visualization
import ipyleaflet
import matplotlib.pyplot as plt
from IPython.display import Image
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score,classification_report,confusion_matrix,precision_score,recall_score

# Planetary Computer Tools
import pystac
import pystac_client
import odc
from pystac_client import Client
from pystac.extensions.eo import EOExtension as eo
from odc.stac import stac_load
import planetary_computer as pc
pc.settings.set_subscription_key('ab0b02101f9d4746a2b082e8135b18bf')

# Others
import requests
import rich.table
from itertools import cycle
from tqdm import tqdm
tqdm.pandas()
#from statistics import mean








from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,ExtraTreesClassifier
from sklearn.ensemble import ( GradientBoostingClassifier,AdaBoostClassifier )
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting train and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
    
            
            models = {
              "Random Forest" :RandomForestClassifier(),
              "SVC" : SVC(),
              "Decision Tree" : DecisionTreeClassifier(),
              "Gradient Boosting" : GradientBoostingClassifier(),
              "Logistic Regression" : LogisticRegression(solver='lbfgs'),
              "XGB Classifier" : XGBClassifier(),
              "AdaBoost Classifier": AdaBoostClassifier(),
              "ExtraTree Classifier": ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, criterion='gini',
                    max_depth=None, max_features='auto', max_leaf_nodes=None,
                    max_samples=None, min_impurity_decrease=0.0, min_samples_leaf=1,
                    min_samples_split=2, min_weight_fraction_leaf=0.0,
                    n_estimators=100, n_jobs=-1, oob_score=False,
                    random_state=42, verbose=0, warm_start=False)   
                
            }
            
            params={
                "Decision Tree": {
                    'criterion':[ 'log_loss', 'entropy', 'gini'],
                     'splitter':['best','random'],
                     'max_features':['sqrt','log2'],
                },
                "SVC" : {
                     'kernel': ['linear', 'rbf'],
                    'C': [0.1, 1, 10]
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                   'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "XGB Classifier":{
                   'learning_rate':[.1,.01,.05,.001],
                   'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Classifier":{
                   'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "ExtraTree Classifier" : {},
                "Logistic Regression" : {}
                
                
                
            }
            
            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params = params)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score  < 0.8:
                raise CustomException("No best model",sys)
            
            logging.info("Found best base model - {0}.".format(best_model_name))
                 
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            acc_score = accuracy_score(predicted,y_test)
            
            return acc_score
            
        except Exception as e:
            raise CustomException(e,sys)
        
        
        
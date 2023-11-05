import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


import os
import warnings
warnings.filterwarnings("ignore")

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_obj(self,numerical,categorical):
        try:
            
            numerical_pipe = Pipeline([
            ('imputer',IterativeImputer()),
            ('imputer1', SimpleImputer(strategy='mean',fill_value=np.NaN)),
            ('scaler', MinMaxScaler())
            ])

            categorical_pipe = Pipeline([

            ('imputer2', SimpleImputer(strategy='most_frequent', fill_value=np.NaN)),
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False))
            ])
            
            logging.info("Numerical and categorical column transformations are done.")
            preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_pipe, numerical),
            ('cat', categorical_pipe, categorical)
            ])
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def  initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path,index_col=False)
            test_df = pd.read_csv(test_path,index_col=False)
            target = 'Class of Land'
            features = train_df.columns
            features = features.drop('Class of Land')
            numerical = train_df[features].select_dtypes('number').columns
            categorical = pd.Index(np.setdiff1d(features, numerical))
            logging.info('Read train and test dataset')
            logging.info('Obtained preprocessing object')
            
            preprocessing_obj = self.get_data_transformer_obj(numerical,categorical)
            
            input_feature_train_df = train_df.drop(columns=['Class of Land'],axis=1)
            target_feature_train_df = train_df['Class of Land']
            
            input_feature_test_df = test_df.drop(columns=["Class of Land"],axis=1)
            target_feature_test_df = test_df['Class of Land']
            
            logging.info("Applying preprocessing object on training and testing data frame")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr , np.array(target_feature_test_df)
            ]
            
            logging.info('Saved preprocessing object.')
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)
            
            
            
        
    



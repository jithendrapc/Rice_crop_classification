import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

import warnings
warnings.filterwarnings('ignore')

# Visualization
import ipyleaflet
import matplotlib.pyplot as plt
from IPython.display import Image
import seaborn as sns

# Data Science
import numpy as np
import pandas as pd

# Feature Engineering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score,classification_report,confusion_matrix,precision_score,recall_score

# Planetary Computer Tools
import pystac
import pystac_client
import odc
from pystac_client import Client
from pystac.extensions.eo import EOExtension as eo
from odc.stac import stac_load
import planetary_computer as pc
pc.settings.set_subscription_key('*****************************')

# Others
import requests
#import rich.table
from itertools import cycle
from tqdm import tqdm
tqdm.pandas()
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            if prediction == 0:
                print("Non Rice")
                return 'Non Rice'
            else:
                print('Rice')
                return 'Rice'
        except Exception as e:
            raise CustomException(e,sys)
    
class CustomData:
    def __init__(self,Latitude,Longitude):
        self.Latitude = Latitude
        self.Longitude = Longitude
        
    
    def get_sentinel_data(self,latlong,season,assests):


        #latlong=latlong.replace('(','').replace(')','').replace(' ','').split(',')
        bbox_of_interest = (float(latlong[1])-(0.0005/2) , float(latlong[0])-(0.0005/2), float(latlong[1])+(0.0005/2) , float(latlong[0])+(0.0005/2))
    
        bands_of_interest = assests
        if season == 'SA':
            time_slice = "2022-05-01/2022-08-31"
        if season == 'WS':
            time_slice = "2022-01-01/2022-04-30"
    
        
        vv_list = []
        vh_list = []
        vv_by_vh_list = []
    
    
        time_of_interest = time_slice
    
        catalog = pystac_client.Client.open(r"https://planetarycomputer.microsoft.com/api/stac/v1")
        search = catalog.search(collections=["sentinel-1-rtc"], bbox=bbox_of_interest, datetime=time_of_interest)
        items = list(search.get_all_items())
        item = items[0]
        items.reverse()
        resolution = 10 
        scale = resolution / 111320.0
        data = stac_load([items[1]],bands=bands_of_interest, patch_url=pc.sign, bbox=bbox_of_interest,crs="EPSG:4326", resolution=scale).isel(time=0)

        for item in items:
            data = stac_load([item], bands=bands_of_interest, patch_url=pc.sign, bbox=bbox_of_interest,crs="EPSG:4326", resolution=scale).isel(time=0)
            if(data['vh'].values[0][0]!=-32768.0 and data['vv'].values[0][0]!=-32768.0):
                data = data.where(~data.isnull(), 0)
                vh = data["vh"].astype("float64")
                vv = data["vv"].astype("float64")
                vv_list.append(np.median(vv))
                vh_list.append(np.median(vh))
                vv_by_vh_list.append(np.median(vv)/np.median(vh))
        logging.info("Sentinal data collection is completed")
              
        return vh_list, vv_list, vv_by_vh_list
    
    
    def get_data_as_data_frame(self):
        try:
            assests = ['vh','vv']
            vh_vv = []
            vh_vv_data=[]
            vh_vv.append(self.get_sentinel_data([float(self.Latitude),float(self.Longitude)],'SA',assests))
            for i in vh_vv[0]:
                vh_vv_data.append(np.array(i).mean())
            print(vh_vv_data)
            vh = vh_vv_data[0]
            vv = vh_vv_data[1]
            vv_by_vh = vh_vv_data[2]
            #print(self.vh,self.vv)
            custom_data_input_dict = {
                "vh":[vh],
                "vv":[vv],
                "vv_by_vh":[vv_by_vh]
            }
            logging.info("Data is collected for selected location")
            
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
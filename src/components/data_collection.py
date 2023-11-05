# Supress Warnings
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
pc.settings.set_subscription_key('ab0b02101f9d4746a2b082e8135b18bf')

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

@dataclass
class DataCollectionConfig:
    collected_data_path: str = os.path.join('artifacts','dataset.csv')
    
class DataCollection:
    def __init__(self):
        self.collection_config = DataCollectionConfig()
        
        
    def get_sentinel_data(self,latlong,season,assets):


        latlong=latlong.replace('(','').replace(')','').replace(' ','').split(',')
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
              
        return vv_list, vh_list, vv_by_vh_list
    
    
    def combine_two_datasets(self,dataset1,dataset2):
        data = pd.concat([dataset1,dataset2], axis=1)
        return data
    
    

    def initiate_data_collection(self):
        logging.info("Entered the data collection process")
        try:
            crop_presence_data = pd.read_csv("Crop_Location_Data_20221201.csv")
            #assests = ['vh','vv']
            #vh_vv = []
            #for coordinates in tqdm(crop_presence_data['Latitude and Longitude']):
                #vh_vv.append(self.get_sentinel_data(coordinates,'SA',assests))
                
            #vh_vv_data = pd.DataFrame(vh_vv,columns =['vh','vv','vv_by_vh'])
            #vh_vv_data.to_csv('vh_vv_data_2022_May_Aug.csv')
            
            vh_vv_data=pd.read_csv("vh_vv_data_2022_May_Aug.csv",index_col=[0])
            crop_data = self.combine_two_datasets(crop_presence_data,vh_vv_data)
            crop_data = crop_data[['vh','vv','vv_by_vh','Class of Land']]   #Removing Latitude Longitude Column
            
            for i in crop_data.columns.drop('Class of Land'):
                for j in range(0,600):
                    crop_data[i][j]=pd.Series(crop_data[i][j].replace('[','').replace(']','').replace(' ','').split(',')).astype('float64') #Converting string to list of float.
                    crop_data[i][j]=crop_data[i][j].mean() 

            os.makedirs(os.path.dirname(self.collection_config.collected_data_path),exist_ok=True)
            crop_data.to_csv(self.collection_config.collected_data_path,index=False,header=True)

            logging.info("Collected dataset as csv file")

        
            
            return  self.collection_config.collected_data_path

            
        except Exception as e:
            raise CustomException(e,sys)    
        
if __name__=="__main__":
    obj=DataCollection()
    collected_data=obj.initiate_data_collection()
    
    data_ingestion = DataIngestion()
    train_data,test_data = data_ingestion.initiate_data_ingestion(collected_data)
    
    
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))
    
    
    

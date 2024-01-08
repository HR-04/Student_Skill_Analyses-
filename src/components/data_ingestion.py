import os
import sys
from src.exception import CustomerException
from src.logger import logging
import pandas as pd


from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_congfig=DataIngestionConfig()
        
    def initiate_data_ingestion(Self):
        logging.info("Entered the data ingestion method or components")
        try:
            df = pd.read_csv(r"F:\ML_Project\notebook\data\stud.csv")
            logging.info("Read the dataset as dataframe")
            
            os.makedirs(os.path.dirname(Self.ingestion_congfig.train_data_path),exist_ok=True)
            
            df.to_csv(Self.ingestion_congfig.raw_data_path,index=False,header=True)
            
            logging.info("Train Test Split Initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(Self.ingestion_congfig.train_data_path,index=False,header=True)
            
            test_set.to_csv(Self.ingestion_congfig.test_data_path,index=False,header=True)
            
            logging.info("Ingestion of the data is completed")
            
            return(
                Self.ingestion_congfig.train_data_path,
                Self.ingestion_congfig.test_data_path
            )
            
        except Exception as e:
            raise CustomerException(e,sys)
        
if __name__ == "__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.componenets.data_ingestion import DataIngestion
from src.componenets.data_transformation import DataTransformation
from src.componenets.model_trainer import ModelTrainer





## to trigger data ingestion
if __name__ == '__main__':
    # create object of DataIngestion class
    obj = DataIngestion()
    # Access method
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)

    # create object of Data Transformation class
    data_transformation = DataTransformation()

    ## This method returns the Preprocessed train and test array, as well as object
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

    ## Create object for model trainer
    model_trainer = ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)
    

    




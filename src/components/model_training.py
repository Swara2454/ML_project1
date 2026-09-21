import os
import sys
from src.exception import CustomException
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from src.components import data_transformation
from src.utils import evaluation_model,save_object
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifact',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            
            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'K-nearestNeighbour regressor':KNeighborsRegressor(),
                'DecisionTree':DecisionTreeRegressor(),
                'RandomForest':RandomForestRegressor(),
                'XGBoost':XGBRegressor(),
                'Catboosting regressor':CatBoostRegressor(verbose=False),
                'Adaboost':AdaBoostRegressor()
            }

            params = {

                'LinearRegression': {},

                'Lasso': {
                    'alpha': [0.001, 0.01, 0.1, 1]
                },

                'Ridge': {
                    'alpha': [0.001, 0.01, 0.1, 1]
                },

                'K-nearestNeighbour regressor': {
                    'n_neighbors': [3, 5, 7, 9]
                },

                'DecisionTree': {
                    'criterion': [
                        'squared_error',
                        'friedman_mse',
                        'absolute_error',
                        'poisson'
                    ]
                },

                'RandomForest': {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                'XGBoost': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                'Catboosting regressor': {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },

                'Adaboost': {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }


            model_report:dict=evaluation_model(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,models=models, param=params)

            best_model_score=max(sorted(model_report.values())) ## to get best model score

            best_model_name =list(model_report.keys())[list(model_report.values()).index(best_model_score)] 

            best_model=models[best_model_name]



            print(best_model)

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info("Best model found")

            save_object(
                file_path=self.model_config.trained_model_file_path,
                obj=best_model
            )

            prediction=best_model.predict(X_test)

            r2=r2_score(y_test,prediction)
            return r2

        except Exception as e:
            raise CustomException(e,sys)



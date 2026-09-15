import sys
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.logger import logging
from src.utils import save_object
import numpy as np


@dataclass
class DataTransformerConfig:
    preprocessor_obj_file_path=os.path.join('artifact',"preprocessor.pkl")

class Datatransformer:
    def __init__(self):
        self.data_transformer_config=DataTransformerConfig()

    def get_transformer_obj(self):
        try:
            numerical_col=['reading score', 'writing score']
            categorical_col=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            num_pipeline=Pipeline(
                steps=
                [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('stdscaler',StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehot',OneHotEncoder()),
                    ('stdscaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical Pipeline created")

            logging.info("Categorical pipeline created")

            preprocessor=ColumnTransformer(
                [('num_pipeline',num_pipeline,numerical_col),
                 ('cat_pipeline',cat_pipeline,categorical_col)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data")

            processing_obj=self.get_transformer_obj()

            target_col='math score'

            input_features_train_df=train_df.drop(columns=[target_col],axis=1)
            target_feature_train_df=train_df[target_col]

            input_features_test_df=test_df.drop(columns=[target_col],axis=1)
            target_feature_test_df=test_df[target_col]

            logging.info('input and target features created')

            input_features_train_arr=processing_obj.fit_transform(input_features_train_df)
            input_features_test_arr=processing_obj.transform(input_features_test_df)

            train_arr= np.c_[
                input_features_train_arr,np.array(target_feature_train_df)
            ]

            
            test_arr=np.c_[
                input_features_test_arr,np.array(target_feature_test_df)
            ]

            logging.info('Saved processing object')

            save_object(
                file_path=self.data_transformer_config.preprocessor_obj_file_path,
                obj=processing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformer_config.preprocessor_obj_file_path
            )



        except Exception as e:
            raise CustomException(e,sys)

    


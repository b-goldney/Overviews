from itertools import chain
from datetime import datetime, timedelta
from typing import Dict, List, Any, TypedDict, Union

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor 
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

from model_grid import MODEL_GRID
from accuracy_scores import calculate_accuracy_scores
from types_ import NumericScaler, ModelGrid, FeatureImportances, ModelsDict, ScoreDict


# CONSTANTS
# Feature Importances

class AutoMl:
    def __init__(self, 
                 df: pd.DataFrame, 
                 num_cols: List[str],
                 cat_cols: List[str],
                 target: List[str],
                 numeric_scaler: NumericScaler = StandardScaler,  
                 categorical_encoder: BaseEstimator = OneHotEncoder,
                 test_size: float = 0.25,
                 model_grid: ModelGrid = MODEL_GRID 
                 ) -> None:
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.x_cols = num_cols + cat_cols
        self.target = target
        self.numeric_scaler = numeric_scaler()
        self.categorical_encoder = categorical_encoder()
        self.test_size = test_size
        self.MODEL_GRID = model_grid

        df = df.dropna(subset=target)
        df[num_cols] = df[num_cols].fillna(0)
        df[cat_cols] = df[cat_cols].fillna('')
        self.df = df
        
    
    def preprocesing(self):
        # train test split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.df, self.df[self.target], test_size=self.test_size)
        
        # convert any categorical variables with nunique == 2 to numeric
        for col in self.cat_cols:
            if self.df[col].nunique() == 2:
                self.cat_cols.remove(col)
                self.num_cols.append(col)
            elif self.df[col].nunique() == 1:
                self.cat_cols.remove(col)

        # create steps for pipeline
        column_transformer = ColumnTransformer(
            transformers = [
                ("standard_scaler", self.numeric_scaler, self.num_cols),
                ("one_hot_encoder", self.categorical_encoder, self.cat_cols)
            ]
        )

        self.feature_selection_ = SelectFromModel(
            ExtraTreesRegressor(n_estimators=50), threshold='median', prefit=False
            )
        self.preprocessing_pipeline = Pipeline([
            ('preprocessing', column_transformer),
            ('feature_selection', self.feature_selection_)
            ])
        self.x_train_ = self.preprocessing_pipeline.fit_transform(
            self.x_train, self.y_train
            )
        self.x_test_ = self.preprocessing_pipeline.transform(self.x_test)
        print(self.x_train.shape, '<<< x_train.shape, \n\n\n')
        print(self.x_train_.shape, '<<< x_train_.shape, \n\n\n')
    
    @property
    def feature_importance_values(self) -> Dict[str, FeatureImportances]:
        ## identify categorical columns after OHE 
        encoded_columns = [x.replace('one_hot_encoder__', '')
                            .replace('standard_scaler__', '') 
                        for x in self.preprocessing_pipeline[0].get_feature_names_out() ] 

        # create dictionary with all feature importances
        feature_importance_values_ = self.feature_selection_.estimator_.feature_importances_ 
        feature_importances = dict(zip(encoded_columns, feature_importance_values_))

        # identify values above the threshold
        threshold_value = np.median(list(feature_importances.values()))
        for key, value in feature_importances.items():
            feature_importances[key] = {
                'value': value,
                'above_threshold': value >= threshold_value 
            }
        return feature_importances
    
    def run_models(self) -> ModelsDict:
        all_models = {}
        for model in self.MODEL_GRID:
            start_time = datetime.now()
            reg = RandomizedSearchCV(model['model'], param_distributions=model['parameter_grid'], 
                                    n_iter=2, cv=2 , refit=True, verbose=0)
            reg.fit(self.x_train_, self.y_train)
            y_predict = reg.predict(self.x_train_)
            scores = calculate_accuracy_scores(self.y_train, y_predict)
            end_time = datetime.now()

            # create final dictionary
            all_models[model['name']] = {} 
            all_models[model['name']]['scores'] = scores
            all_models[model['name']]['model'] = reg
            all_models[model['name']]['time_to_complete'] = end_time - start_time
            print(f'completed {model["name"]} @ {datetime.now()}')
        self.all_models_  = all_models
        return all_models
    
    def _get_original_column_names(self):
        get_support = self.preprocessing_pipeline.named_steps['feature_selection'].get_support()
        feature_names_out = self.preprocessing_pipeline.named_steps['preprocessing'].get_feature_names_out()
        final_encoded_columns = [s for s, b in zip(feature_names_out, get_support) if b]
        final_encoded_columns

        original_columns_after_fs = {
            'numeric_columns': [],
            'categorical_columns': []
        }
        for col in final_encoded_columns:
            if 'standard_scaler__' in col:
                original_columns_after_fs['numeric_columns'].append(col.replace('standard_scaler__',''))
            elif 'one_hot_encoder__' in col:
                temp_name = col.replace('one_hot_encoder__','')
                original_col_name = [x for x in self.cat_cols if temp_name.startswith(x)]
                if original_col_name[0] not in original_columns_after_fs['categorical_columns']:
                    original_columns_after_fs['categorical_columns'].append(original_col_name[0])
        self.original_columns_after_fs = original_columns_after_fs
        self.encoded_column_names_after_fs = final_encoded_columns
        return original_columns_after_fs
    
    @property
    def get_predictions(self):
        self._get_original_column_names()

        # identify the best model
        best_score = 1e100000
        best_model = None
        for model in self.all_models_.items():
            if model[1]['scores']['mae'] < best_score:
                best_score = model[1]['scores']['mae']
                best_model= model[1]['model']
        
        # create final df (with original data)
        self.train = pd.DataFrame(self.x_train).reset_index(drop=True)
        self.test = pd.DataFrame(self.x_test).reset_index(drop=True)
        # create final df (with preprocessed data)
        self.train_ = pd.DataFrame(self.x_train_).reset_index(drop=True)
        self.test_ = pd.DataFrame(self.x_test_).reset_index(drop=True)

        # add actual value
        # original data
        self.train[self.target] = self.y_train
        self.test[self.target] = self.y_test
        # processed data
        self.train_[self.target] = self.y_train
        self.test_[self.target] = self.y_test

        # add prediction
        train_predict = best_model.predict(self.x_train_)
        test_predict = best_model.predict(self.x_test_)
        # original data
        self.train['y_predict'] = train_predict
        self.test['y_predict'] = test_predict

        # processed data
        self.train_['y_predict'] = train_predict
        self.test_['y_predict'] = test_predict

        # train test id
        # original data
        self.train['train_test_id'] = 'train'
        self.test['train_test_id'] = 'test'
        # processed data
        self.train_['train_test_id'] = 'train'
        self.test_['train_test_id'] = 'test'

        # update column names
        self.train_.columns = self.encoded_column_names_after_fs + [self.target[0], 'y_predict', 'train']
        self.test_.columns = self.encoded_column_names_after_fs + [self.target[0], 'y_predict', 'train']

        return { 
            'original_data' : pd.concat([self.train, self.test]) ,
            'processed_data' : pd.concat([self.train_, self.test_]) 
            }

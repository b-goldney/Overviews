from typing import Dict, List, Any, Union
import os
from itertools import compress, groupby
import random
from operator import itemgetter
import time
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from .accuracy_metrics import accuracy_metrics
from .predictions import predictions
from ads.utils.encoder import NpEncoder
from ads.utils.sql import update_job_ml_completed_jobs
from ads.utils.logger import logger
from ads.aml.helpers import get_scaler, get_imputer
from .user_input import UserInput
from .enums import ProblemType


class AutoMl:
    def __init__(self, df: pd.DataFrame, user_input):
        self.df = df
        self.user_input = UserInput(**user_input)  # create_user_input()
        self.models = self.user_input.models

        # columns
        self.numeric_columns = self.user_input.numericColumns
        self.categorical_columns = self.user_input.categoricalColumns
        self.target = self.user_input.targetColumn
        self.x_columns = self.numeric_columns + self.categorical_columns

    def train_test_split(self) -> None:
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.df[self.x_columns],
            self.df[self.target],
            test_size=float(self.user_input.test_size),
            random_state=42,
            shuffle=self.user_input.shuffle,
        )

    def _create_pipeline(self):
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", get_imputer(self.user_input)),
                ("scaler", get_scaler(self.user_input)),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_columns),
                ("cat", categorical_transformer, self.categorical_columns),
            ]
        )

        # Define preprocessing and feature selection pipeline
        if self.user_input.feature_selection is True:
            self.feature_selection = SelectFromModel(
                RandomForestRegressor(n_estimators=100), threshold="median"
            )
            self.pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("feature_selection", self.feature_selection),
                ]
            )
        else:
            self.pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    def preprocess(self):
        self._create_pipeline()
        self.x_preprocessed_train = self.pipeline.fit_transform(
            self.x_train, self.y_train
        )
        self.x_preprocessed_test = self.pipeline.transform(self.x_test)

    def model_selection(self):
        # Define the desired number of parameter samples
        n_samples = self.user_input.n_param_samples

        # Define a list to store the results
        model_results = []

        # Iterate over each model
        best_score = -1e10
        counter = 0
        for model in self.models:
            counter += 1
            estimator = model["model"]
            param_grid = model["parameter_grid"]
            start_time = time.time()  # Start the timer

            # ensure that n_samples is not greater than the max number
            # of possible samples
            max_num_samples = np.prod([len(values) for values in param_grid.values()])
            n_samples = min(n_samples, int(max_num_samples))
            if os.environ.get("LOCAL_DEVELOPMENT"):
                n_samples = 2

            print(
                f"\n\n-------------------- n_samples set to {n_samples} -------------------- \n\n",
                flush=True,
            )
            # Perform multiple fits with different parameter samples
            for _ in range(int(n_samples)):
                # Randomly sample the parameter grid
                param_samples = {
                    param: random.choice(values) for param, values in param_grid.items()
                }

                # Set the model parameters
                estimator.set_params(**param_samples)

                # Calculate the cross-validated scores
                try:
                    scores = cross_val_score(
                        estimator,
                        self.x_preprocessed_train,
                        self.y_train,
                        cv=2
                        if os.environ.get("LOCAL_DEVELOPMENT")
                        else int(self.user_input.k_fold),
                        n_jobs=4,
                        verbose=3,
                        error_score=0,
                    )
                    mean_score = scores.mean()
                except Exception as e:
                    mean_score = 0
                    scores = [0, 0]
                    print(f"Error from cross_val_score: {e}", flush=True)
                if mean_score > best_score:
                    self.best_model = estimator
                    self.best_model.set_params(**param_samples)
                    best_score = mean_score

                # Store the results in a dictionary
                elapsed_time = format(round(time.time() - start_time, 2), ".2f")
                model_result = {
                    "model": str(model["name"]),
                    "params": {
                        key: str(value) for key, value in param_samples.items()
                    },  # convert all values to strings
                    "scores": [str(x) for x in scores],
                    "mean_score": mean_score,
                    "time_to_fit": str(elapsed_time),
                }
                # logger.info(json.dumps(model_result), "<<< model_result \n\n\n")

                # Append the result to the list then sort by mean_score and time_to_fit
                model_results.append(model_result)

                # Group by 'model' and select the best based on mean_score
                model_results = [
                    max(list(group), key=itemgetter("mean_score"))
                    for _, group in groupby(
                        sorted(model_results, key=itemgetter("model")),
                        key=itemgetter("model"),
                    )
                ]

                # Now sort by mean_score and time_to_fit
                model_results = sorted(
                    model_results,
                    key=itemgetter("mean_score", "time_to_fit"),
                    reverse=True,
                )
                # model_results = sorted(
                #    model_results,
                #    key=itemgetter("mean_score", "time_to_fit"),
                #    reverse=True,
                # )

                # update the DB
                update_job_ml_completed_jobs(
                    is_running="t",
                    job_was_successful="f",
                    completed_jobs=json.dumps(model_results, cls=NpEncoder),
                )

        # fit the best model and set params
        if hasattr(self, "best_model"):
            self.best_model.fit(self.x_preprocessed_train, self.y_train)
        else:
            Exception("No best model found")

        # save key results
        for key in model_results:
            # key["params"] = json.dumps(key["params"], cls=NpEncoder)
            self.model_results_ = model_results

    def predict(self):
        self.y_predict_train = self.best_model.predict(self.x_preprocessed_train)
        self.y_predict_test = self.best_model.predict(self.x_preprocessed_test)

    def get_column_transformations(self):
        self.one_hot = (
            self.pipeline.named_steps["preprocessor"]
            .transformers_[1][1]
            .named_steps["onehot"]
        )

        # column transformations
        self.ct_cat_feature_names_out_ = list(self.one_hot.get_feature_names_out())
        self.ct_feature_names_out_ = (
            self.numeric_columns + self.ct_cat_feature_names_out_
        )

    def get_feature_selection_output(self):
        # get all feature names from column transformations
        if self.user_input.feature_selection is True:
            # get column names that were selected by the feature selection step
            self.fs_selected_features_out_ = list(
                compress(
                    self.ct_feature_names_out_,
                    self.pipeline.named_steps["feature_selection"].get_support(),
                )
            )

            # combine feature importance values with feature names and sort in descending order
            feature_importance = self.pipeline.named_steps["feature_selection"]
            dict_combined = dict(
                zip(
                    self.ct_feature_names_out_,
                    feature_importance.estimator_.feature_importances_,
                )
            )
            self.fs_feature_importances_ = dict(
                sorted(dict_combined.items(), key=lambda item: item[1], reverse=True)
            )

            # map the encoded column names back to the original column names
            # Step 1: create a dictionary with every categorical column and all it's unique values
            # Step 2: recreate OHE's column names (i.e. <column_name>_<category>),
            # Step 3: match step 2 values against values in step 1
            fs_encoded_cat_columns = [
                x
                for x in self.fs_selected_features_out_
                if x not in self.numeric_columns
            ]

            temp = dict(zip(self.one_hot.feature_names_in_, self.one_hot.categories_))
            for key, value in temp.items():
                new_values = [f"{key}_{v}" for v in value]
                temp[key] = new_values
            self.fs_selected_original_col_names_out = [
                key
                for key, value in temp.items()
                if any(val in fs_encoded_cat_columns for val in value)
            ]

    def trim_df(self):
        # capture x_cols, based off feature_selection
        if self.user_input.feature_selection is True:
            x_cols = self.fs_selected_features_out_
        else:
            x_cols = self.ct_feature_names_out_

        # create trim DF
        x_trim_train = pd.DataFrame(self.x_preprocessed_train, columns=x_cols)
        y_trim_train = pd.DataFrame(self.y_predict_train, columns=["predicted"]).fillna(
            0
        )
        y_trim_train_true = pd.DataFrame(self.y_train).reset_index(drop=True)
        y_trim_train_true.columns = ["actual"]
        trim_train_df = pd.concat(
            [x_trim_train, y_trim_train_true, y_trim_train], axis=1
        )
        trim_train_df["train_test_id"] = "train"

        x_trim_test = pd.DataFrame(self.x_preprocessed_test, columns=x_cols)
        y_trim_test = pd.DataFrame(self.y_predict_test, columns=["predicted"]).fillna(0)
        y_trim_test_true = pd.DataFrame(self.y_test).reset_index(drop=True)
        y_trim_test_true.columns = ["actual"]
        trim_test_df = pd.concat([x_trim_test, y_trim_test, y_trim_test_true], axis=1)
        trim_test_df["train_test_id"] = "test"
        self.trim_df_ = (
            pd.concat([trim_train_df, trim_test_df], axis=0)
            .fillna(0)
            .reset_index(drop=True)
        )

    def run_all(self):
        self.train_test_split()
        self.preprocess()
        self.model_selection()
        self.predict()
        self.get_column_transformations()
        self.get_feature_selection_output()
        self.trim_df()


def run_auto_ml(df: pd.DataFrame, user_input):
    # fit model
    auto_ml = AutoMl(df, user_input)
    auto_ml.run_all()

    # create output
    trim_train_df = auto_ml.trim_df_[auto_ml.trim_df_["train_test_id"] == "train"]
    trim_test_df = auto_ml.trim_df_[auto_ml.trim_df_["train_test_id"] == "test"]
    accuracy_metrics_ = accuracy_metrics(
        auto_ml.user_input.problemType, trim_train_df, trim_test_df
    )
    overview_ = {
        "problem_type": auto_ml.user_input.problemType,
        "accuracy_metrics": accuracy_metrics_,
        "best_estimator": type(auto_ml.best_model).__name__,
        "best_parameters": auto_ml.model_results_[0]["params"],
        "model_history": auto_ml.model_results_,
        "numeric_columns": auto_ml.numeric_columns,
        "categorical_columns": auto_ml.categorical_columns,
        "preprocessed_columns": list(auto_ml.ct_feature_names_out_),
    }

    # NOTE: these categories (e.g. overview, predictions, table) match
    # the tabs on the frontend for "analyze output" page. The 'Table'
    # section is output in a separate section, which speeds up load
    # times to the frontend.
    accuracy_metrics_ = {
        "overview": overview_,
        "predictions": predictions(
            auto_ml.user_input.problemType, trim_train_df, trim_test_df
        ),
        "feature_selection": auto_ml.fs_feature_importances_
        if auto_ml.user_input.feature_selection is True
        else "skipped",
        "model_explainability": "",
        "production": "",
        "inputs": auto_ml.user_input.to_dict(),
    }

    return auto_ml, accuracy_metrics_

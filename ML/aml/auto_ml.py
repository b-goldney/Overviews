from typing import Dict, List, Any, TypedDict, Union
import os
from datetime import datetime
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score


from .user_input import UserInput
from .types_ import ModelsDict
from .accuracy_metrics import accuracy_metrics
from .predictions import predictions
from ads.utils.sql import update_job_ml_completed_jobs
from ads.utils.encoder import NpEncoder
from .model_explainability import calculate_shap_values
from .enums import ProblemType


class IFeatureImportance(TypedDict):
    column: str
    value: float
    above_threshold: str


class AutoMl:
    def __init__(self, df: pd.DataFrame, user_input: UserInput) -> None:
        self.num_cols = user_input.numericColumns
        self.cat_cols = user_input.categoricalColumns
        self.x_cols = user_input.numericColumns + user_input.categoricalColumns
        self.target = user_input.targetColumn
        self.numeric_scaler = StandardScaler()
        self.categorical_encoder = OneHotEncoder()
        self.test_size = 0.25
        self.user_input = user_input
        self.MODEL_GRID = user_input.parameter_grid
        print(df.columns, "<<<< df.columns \n\n\n", flush=True)

        df = df.dropna(subset=user_input.targetColumn)
        df[user_input.numericColumns] = df[user_input.numericColumns].fillna(0)
        df[user_input.categoricalColumns] = df[user_input.categoricalColumns].fillna("")
        self.df = df

    def preprocesing(self):
        # train test split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.df, self.df[self.target], test_size=self.test_size
        )

        # convert any categorical variables with nunique == 2 to numeric
        # NOTE: this doesn't work when the categorical columns are strings
        # like "yes" and "no" because we can't convert strings to numbers
        # for col in self.cat_cols:
        #    if self.df[col].nunique() == 2:
        #        self.cat_cols.remove(col)
        #        self.num_cols.append(col)
        #    elif self.df[col].nunique() == 1:
        #        self.cat_cols.remove(col)

        # create steps for pipeline
        column_transformer = ColumnTransformer(
            transformers=[
                ("standard_scaler", self.numeric_scaler, self.num_cols),
                ("one_hot_encoder", self.categorical_encoder, self.cat_cols),
            ]
        )

        self.feature_selection_ = SelectFromModel(
            ExtraTreesRegressor(n_estimators=50), threshold="median", prefit=False
        )
        self.preprocessing_pipeline = Pipeline(
            [
                ("preprocessing", column_transformer),
                ("feature_selection", self.feature_selection_),
            ]
        )
        self.x_train_ = self.preprocessing_pipeline.fit_transform(
            self.x_train, self.y_train
        )
        self.x_test_ = self.preprocessing_pipeline.transform(self.x_test)

        # self.x_train_.columns = [
        #    name.replace("standard_scaler__", "").replace("one_hot_encoder__", "")
        #    for name in self.x_train_.columns
        # ]
        # self.x_test_.columns = [
        #    name.replace("standard_scaler__", "").replace("one_hot_encoder__", "")
        #    for name in self.x_test_.columns
        # ]

    @property
    def feature_importance_values(self) -> List[IFeatureImportance]:
        # identify categorical columns after OHE
        encoded_columns = [
            x.replace("one_hot_encoder__", "").replace("standard_scaler__", "")
            for x in self.preprocessing_pipeline[0].get_feature_names_out()
        ]

        # create dictionary with all feature importances
        feature_importance_values_ = (
            self.feature_selection_.estimator_.feature_importances_
        )
        feature_importances = dict(zip(encoded_columns, feature_importance_values_))

        # identify values above the threshold
        threshold_value = np.median(list(feature_importances.values()))
        for key, value in feature_importances.items():
            feature_importances[key] = {
                "value": value,
                "above_threshold": value >= threshold_value,
            }
        # convert from dictionary to list of dictionaries
        feature_importances = [
            {"column": key, **value} for key, value in feature_importances.items()
        ]
        feature_importances = sorted(
            feature_importances, key=lambda x: x["value"], reverse=True
        )

        return feature_importances

    def _evaluate_model(self, actuals, predictions):
        if (
            self.user_input.problemType == ProblemType.REGRESSION.value
            and self.user_input.custom_loss_function is None
        ):
            return mean_absolute_error(actuals, predictions)
        elif (
            self.user_input.problemType == ProblemType.REGRESSION.value
            and self.user_input.custom_loss_function is not None
        ):
            return self.user_input.custom_loss_function(actuals, predictions)
        elif self.user_input.problemType == ProblemType.CLASSIFICATION.value:
            return accuracy_score(actuals, predictions)
        else:
            return accuracy_score(actuals, predictions)

    def run_models(self) -> ModelsDict:
        all_models = []
        n_iter = 2 if os.environ.get("LOCAL_DEVELOPMENT") else 5
        cv = 2 if os.environ.get("LOCAL_DEVELOPMENT") else 5
        best_score = np.inf
        for model in self.MODEL_GRID:
            start_time = datetime.now()
            reg = RandomizedSearchCV(
                model["model"],
                param_distributions=model["parameter_grid"],
                n_iter=n_iter,
                cv=cv,
                refit=True,
                verbose=0,
            )
            reg.fit(self.x_train_, self.y_train)
            y_predict = reg.predict(self.x_train_)
            end_time = datetime.now()
            time_difference = end_time - start_time

            # NOTE: r2_score is used to identify the best model, mae_score
            # is sent to the frontnend for the user. R2 score is easier
            # to identify highest/lowest score since 1 is always the best score
            score_from_model = self._evaluate_model(self.y_train, y_predict)

            if score_from_model < best_score:
                best_score = score_from_model
                best_model = reg

            # create final dictionary
            current_model = {}
            current_model["Name"] = model["full_name"]
            current_model["Accuracy"] = {}
            current_model["Accuracy"]["metric"] = self.user_input.accuracy_metric
            current_model["Accuracy"]["score"] = score_from_model
            current_model["Best Parameters"] = reg.best_params_
            current_model["Time to Complete (seconds)"] = round(
                time_difference.total_seconds(), 1
            )
            all_models.append(current_model)
            print(f'completed {model["name"]} @ {datetime.now()}')

            # update the database
            model_results = sorted(
                all_models,
                key=lambda x: (
                    x["Accuracy"]["score"],
                    x["Time to Complete (seconds)"],
                ),
                reverse=self.user_input.sort_direction_for_accuracy_metric,
            )

            update_job_ml_completed_jobs(
                is_running="t",
                job_was_successful="f",
                completed_jobs=json.dumps(model_results, cls=NpEncoder),
            )

        self.all_models_ = all_models
        self.best_model = best_model
        return all_models

    def _get_original_column_names(self):
        get_support = self.preprocessing_pipeline.named_steps[
            "feature_selection"
        ].get_support()
        feature_names_out = self.preprocessing_pipeline.named_steps[
            "preprocessing"
        ].get_feature_names_out()
        final_encoded_columns = [s for s, b in zip(feature_names_out, get_support) if b]
        self.encoded_column_names_after_fs = [
            name.replace("standard_scaler__", "").replace("one_hot_encoder__", "")
            for name in final_encoded_columns
        ]

        # original_columns_after_fs = {"numeric_columns": [], "categorical_columns": []}
        # for col in final_encoded_columns:
        #    if "standard_scaler__" in col:
        #        original_columns_after_fs["numeric_columns"].append(
        #            col.replace("standard_scaler__", "")
        #        )
        #    elif "one_hot_encoder__" in col:
        #        temp_name = col.replace("one_hot_encoder__", "")
        #        original_col_name = [
        #            x for x in self.cat_cols if temp_name.startswith(x)
        #        ]
        #        if (
        #            original_col_name[0]
        #            not in original_columns_after_fs["categorical_columns"]
        #        ):
        #            original_columns_after_fs["categorical_columns"].append(
        #                original_col_name[0]
        #            )
        # self.original_columns_after_fs = original_columns_after_fs
        # self.encoded_column_names_after_fs = final_encoded_columns
        # return original_columns_after_fs

    def get_best_model(self):
        return min(self.all_models_.items(), key=lambda x: x[1]["mae_score"])[1][
            "model"
        ]

    def prepare_dataframe(self, x_data, y_data, prediction, train_test_id):
        df = pd.DataFrame(x_data).reset_index(drop=True)
        df[self.target] = y_data.reset_index(drop=True)
        df["y_predict"] = prediction
        df["train_test_id"] = train_test_id
        return df

    @property
    def get_predictions(self):
        self._get_original_column_names()

        # identify the best model
        # self.best_model = self.get_best_model()

        # Predictions for train and test data
        train_predict = self.best_model.predict(self.x_train_)
        test_predict = self.best_model.predict(self.x_test_)

        # Create and populate dataframes for original and processed data
        self.train = self.prepare_dataframe(
            self.x_train, self.y_train, train_predict, "train"
        )
        self.test = self.prepare_dataframe(
            self.x_test, self.y_test, test_predict, "test"
        )
        self.train_ = self.prepare_dataframe(
            self.x_train_, self.y_train, train_predict, "train"
        )
        self.test_ = self.prepare_dataframe(
            self.x_test_, self.y_test, test_predict, "test"
        )

        # Update column names for processed data
        columns = self.encoded_column_names_after_fs + [
            self.target,
            "y_predict",
            "train_test_id",
        ]
        self.train_.columns = columns
        self.test_.columns = columns

        return {
            "original_data": pd.concat([self.train, self.test]).reset_index(drop=True),
            "processed_data": pd.concat([self.train_, self.test_]).reset_index(
                drop=True
            ),
        }

    def shap_values(self):
        self.shap_plot_buffer = calculate_shap_values(
            self.best_model,
            self.get_predictions["processed_data"]
            .drop(columns=["y_predict", "train_test_id", self.target])
            .reset_index(drop=True)
            .loc[0:25, :],
        )


def run_auto_ml(df: pd.DataFrame, user_input: UserInput) -> AutoMl:
    print(user_input, "<<< user_input from auto_ml \n\n\n", flush=True)
    user_input = UserInput(**user_input)  # create_user_input()
    auto_ml = AutoMl(df, user_input)
    auto_ml.preprocesing()
    auto_ml.run_models()
    auto_ml.shap_values()
    auto_ml.get_predictions

    # calculate accuracy metrics
    df = auto_ml.get_predictions["processed_data"]
    train_df = df[df["train_test_id"] == "train"].reset_index(drop=True)
    test_df = df[df["train_test_id"] == "test"].reset_index(drop=True)
    accuracy_metrics_ = accuracy_metrics(
        user_input.problemType, user_input.targetColumn, train_df, test_df
    )
    # create output
    overview_ = {
        "problem_type": user_input.problemType,
        "accuracy_metrics": accuracy_metrics_,
        "best_estimator": auto_ml.best_model.best_estimator_.__class__.__name__,
        "best_parameters": auto_ml.all_models_,  # auto_ml.model_results_[0]["params"],
        "model_history": auto_ml.all_models_,  # auto_ml.model_results_,
        "numeric_columns": user_input.numericColumns,
        "categorical_columns": user_input.categoricalColumns,
        "preprocessed_columns": list(
            auto_ml.preprocessing_pipeline[0].get_feature_names_out()
        ),
    }

    # NOTE: these categories (e.g. overview, predictions, table) match
    # the tabs on the frontend for "analyze output" page. The 'Table'
    # section is output in a separate section, which speeds up load
    # times to the frontend.
    model_output = {
        "overview": overview_,
        "predictions": predictions(
            user_input.problemType, user_input.targetColumn, train_df, test_df
        ),
        "feature_selection": auto_ml.feature_importance_values
        if user_input.feature_selection is True
        else "skipped",
        "model_explainability": "",
        "production": "",
        "inputs": user_input.to_dict(),
    }
    return auto_ml, model_output

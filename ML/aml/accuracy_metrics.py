from typing import List, Tuple, Dict, Union, Literal

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)


def accuracy_metrics(
    problem_type: Literal["regression", "classification"],
    target: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> List[Dict[str, str | float]]:
    if problem_type == "regression":
        return regression_accuracy_metrics(train_df, test_df, target)
    if problem_type == "classification":
        return classification_accuracy_metrics(train_df, test_df, target)


def _classification_accuracy_metrics(
    df: pd.DataFrame, target: str
) -> Tuple[Dict[str, float], Dict[str, int]]:
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(df[target], df.y_predict),
        "precision": precision_score(df[target], df.y_predict, average="macro"),
        "recall": recall_score(df[target], df.y_predict, average="macro"),
        "f1_score": f1_score(df[target], df.y_predict, average="macro"),
    }

    tn, fp, fn, tp = confusion_matrix(df[target], df.y_predict).ravel()
    confusion_matrix_data: Dict[str, int] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

    return metrics, confusion_matrix_data


def classification_accuracy_metrics(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target: str
) -> List[Dict[str, Union[str, float]]]:
    train_metrics, confusion_matrix_train = _classification_accuracy_metrics(
        train_df, target
    )
    test_metrics, confusion_matrix_test = _classification_accuracy_metrics(
        test_df, target
    )

    return [
        {
            "key": "Precision",
            "train": train_metrics["precision"],
            "test": test_metrics["precision"],
        },
        {
            "key": "Recall",
            "train": train_metrics["recall"],
            "test": test_metrics["recall"],
        },
        {
            "key": "F1 Score",
            "train": train_metrics["f1_score"],
            "test": test_metrics["f1_score"],
        },
        {
            "key": "Accuracy",
            "train": train_metrics["accuracy"],
            "test": test_metrics["accuracy"],
        },
        {
            "key": "Confusion Matrix",
            "train": confusion_matrix_train,
            "test": confusion_matrix_test,
        },
    ]


def regression_accuracy_metrics(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target: str
) -> List[Dict[str, str | float]]:
    return [
        {
            "key": "R2",
            "train": r2_score(train_df[target], train_df.y_predict),
            "test": r2_score(test_df[target], test_df.y_predict),
        },
        {
            "key": "Mean Absolute Error",
            "train": mean_absolute_error(train_df[target], train_df.y_predict),
            "test": mean_absolute_error(test_df[target], test_df.y_predict),
        },
        {
            "key": "MAPE",
            "train": 100
            * mean_absolute_percentage_error(train_df[target], train_df.y_predict),
            "test": 100
            * mean_absolute_percentage_error(test_df[target], test_df.y_predict),
        },
    ]

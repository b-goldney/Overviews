from typing import Any, List, Literal, Dict

import pandas as pd
import numpy as np


def format_bin(bin_str):
    # Remove brackets
    bin_str = (
        bin_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    )

    # Split the string into two parts
    start, end = bin_str.split(",")

    # Convert to float, format, and then convert back to string
    start = float(start.strip())
    end = float(end.strip())

    # Format with or without decimals based on value
    start_str = f"{start:,.0f}" if abs(start) >= 10 else f"{start:,.2f}"
    end_str = f"{end:,.0f}" if abs(end) >= 10 else f"{end:,.2f}"
    return f"{start_str} - {end_str}"


def calculate_error_percentage_histogram(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df.loc[:, "error_percentage"] = (df.y_predict - df[target]) / df[target].replace(
        0, np.nan
    )

    hist = pd.DataFrame(
        pd.cut(df.error_percentage, bins=25).value_counts().reset_index()
    )
    hist.columns = ["bins", "count"]
    hist = hist.sort_values("bins").reset_index(drop=True)

    # identify the starting and ending point for each bin
    hist["end_bin"] = hist.bins.apply(lambda x: str(x.right)).values
    hist["start_bin"] = hist.bins.apply(lambda x: str(x.left)).values
    hist["bins"] = hist["bins"].astype(str).apply(format_bin)

    hist["bins"] = hist.start_bin.astype(str) + " - " + hist.end_bin.astype(str)

    return hist[["bins", "count"]].to_dict()


def calculate_value_distribution_histogram(
    df: pd.DataFrame, target: str, num_bins: int = 25
) -> pd.DataFrame:
    # Define the range for the bins
    min_edge = min(df[target].min(), df["y_predict"].min())
    max_edge = max(df[target].max(), df["y_predict"].max())
    bins = np.linspace(min_edge, max_edge, num_bins + 1)

    # Create histograms for actual and predicted values
    actual_hist = (
        pd.cut(df[target], bins=bins, include_lowest=True).value_counts().reset_index()
    )
    predicted_hist = (
        pd.cut(df["y_predict"], bins=bins, include_lowest=True)
        .value_counts()
        .reset_index()
    )

    # Rename columns for clarity
    actual_hist.columns = ["bins", "actual_count"]
    predicted_hist.columns = ["bins", "predicted_count"]

    # Sort the dataframes based on bins
    actual_hist = actual_hist.sort_values("bins").reset_index(drop=True)
    predicted_hist = predicted_hist.sort_values("bins").reset_index(drop=True)

    # Combine the histograms
    combined_hist = pd.merge(
        actual_hist, predicted_hist, on="bins", how="outer"
    ).fillna(0)

    # Format the bins as strings for display
    combined_hist["bins"] = combined_hist["bins"].astype(str).apply(format_bin)

    return combined_hist[["bins", "actual_count", "predicted_count"]].to_dict()


def predictions(
    problem_type: Literal["regression", "classification"],
    target: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> List[Dict[str, str | float]]:
    if problem_type == "regression":
        return _predictions_regression(target, train_df, test_df)
    if problem_type == "classification":
        return _predictions_classification(target, train_df, test_df)


def _predictions_regression(
    target: str, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Dict[str, Any]:
    # summary numbers
    summary_numbers_train = {
        "average_actual": train_df[target].mean(),
        "average_predicted": train_df.y_predict.mean(),
        "median_actual": train_df[target].median(),
        "median_predicted": train_df.y_predict.median(),
        "min_actual": train_df[target].min(),
        "min_predicted": train_df.y_predict.min(),
        "max_actual": train_df[target].max(),
        "max_predicted": train_df.y_predict.max(),
    }
    summary_numbers_test = {
        "average_actual": test_df[target].mean(),
        "average_predicted": test_df.y_predict.mean(),
        "median_actual": test_df[target].median(),
        "median_predicted": test_df.y_predict.median(),
        "min_actual": test_df[target].min(),
        "min_predicted": test_df.y_predict.min(),
        "max_actual": test_df[target].max(),
        "max_predicted": test_df.y_predict.max(),
    }

    error_percentage_histgoram_train = calculate_error_percentage_histogram(
        train_df, target
    )
    error_percentage_histogram_test = calculate_error_percentage_histogram(
        test_df, target
    )
    predictions_histgoram_train = calculate_value_distribution_histogram(
        train_df, target
    )
    predictions_histogram_test = calculate_value_distribution_histogram(test_df, target)
    return {
        "error_percentage_histogram_train": error_percentage_histgoram_train,
        "error_percentage_histogram_test": error_percentage_histogram_test,
        "predictions_histogram_train": predictions_histgoram_train,
        "predictions_histogram_test": predictions_histogram_test,
        "summary_numbers_train": summary_numbers_train,
        "summary_numbers_test": summary_numbers_test,
    }


def _predictions_classification(
    target: str, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Dict[str, Any]:
    # train_metrics = {
    #    "class_distribution": train_df[target].value_counts().to_dict(),
    # }

    # test_metrics = {
    #    "class_distribution": test_df[target].value_counts().to_dict(),
    # }

    # return train_metrics, test_metrics
    return {}, {}

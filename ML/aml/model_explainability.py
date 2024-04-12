import io

import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.base import BaseEstimator


def calculate_shap_values(model: BaseEstimator, x_data: pd.DataFrame) -> None:
    # Create a SHAP explainer and compute SHAP values
    # x_data = pd.DataFrame(x_data, columns=column_names)
    # ex = shap.KernelExplainer(model.predict, x_data)
    # return ex.shap_values(x_data)
    ex = shap.KernelExplainer(model.predict, x_data)
    shap_values = ex.shap_values(x_data)

    # Create a matplotlib figure
    # plt.figure()

    # Generate the SHAP summary plot
    shap.summary_plot(shap_values, x_data, show=False)

    # Save the figure to a BytesIO buffer instead of a file
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)  # Move to the beginning of the buffer

    # Store the buffer in the class attribute
    plt.close()
    return buffer

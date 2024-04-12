from dataclasses import dataclass, asdict, field
import json
import os
from typing import List, Literal, Optional, Callable
from .model_grid import REGRESSION_PARAMETER_GRID, CLASSIFICATION_PARAMETER_GRID


def convert_string_to_json_array(input_string: str) -> str:
    # Split the string into individual values
    values = input_string.split(",")

    # Create a list from the extracted values
    value_list = [value.strip().replace('"', "") for value in values]

    # Convert the list to a JSON array
    json_array = json.dumps(value_list)

    return json_array


def camel_to_snake(string: str) -> str:
    return "".join(["_" + c.lower() if c.isupper() else c for c in string]).lstrip("_")


# Update the UserInput dataclass to use the Enums
@dataclass
class UserInput:
    # meta
    problemType: Literal["regression", "classification"] = "regression"
    numericColumns: List[str] = field(default_factory=list)
    categoricalColumns: List[str] = field(default_factory=list)
    targetColumn: List[str] = None
    workflow: str = "autoMl"
    model_name: Optional[str] = os.environ.get("MODEL_NAME")

    # train test split parameters
    test_size: float = 0.2
    shuffle: bool = True

    # processing
    scaler: Literal["standard", "min-max"] = "standard"

    # model parameters
    n_iter: int = 50
    k_fold: int = 5
    n_jobs: int = -1
    imputer: Literal["simple", "knn"] = "simple"
    imputer_strategy: Literal["mean", "median", "constant"] = "mean"
    imputer_fill_value: int = 0
    n_param_samples: int = 5
    feature_selection: bool = True

    # cusotom
    custom_loss_function: Callable = None

    def __post_init__(self):
        if self.problemType == "regression":
            self.parameter_grid = REGRESSION_PARAMETER_GRID
            self.sort_direction_for_accuracy_metric = False
        elif self.problemType == "classification":
            self.parameter_grid = CLASSIFICATION_PARAMETER_GRID
            self.sort_direction_for_accuracy_metric = True
        else:
            raise ValueError(
                f"problemType should be either 'regression' or 'classification', not {self.problemType}"
            )

        # set accuracy metric
        if self.problemType == "regression" and self.custom_loss_function is None:
            self.accuracy_metric = "Mean Absolute Error"
        elif self.problemType == "regression" and self.custom_loss_function is not None:
            self.accuracy_metric = "Custom"
        else:
            self.accuracy_metric = "Accuracy"

        # update feature_selection
        # if self.feature_selection == "true":
        #    self.feature_selection = True
        # else:
        #    self.feature_selection = False

    def to_dict(self):
        return asdict(self)

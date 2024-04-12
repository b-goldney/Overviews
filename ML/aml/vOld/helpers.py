from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer

from .user_input import UserInput


def get_scaler(user_input: UserInput):
    scaler = user_input.scaler
    if scaler == "standard":
        return StandardScaler()
    elif scaler == "minmax":
        return MinMaxScaler()
    else:
        raise ValueError("Invalid scaler option. Choose 'standard' or 'minmax'.")


def get_imputer(user_input: UserInput):
    imputer = user_input.imputer
    imputer_strategy = user_input.imputer_strategy

    if imputer == "simple" and imputer_strategy != "constant":
        return SimpleImputer(strategy=imputer_strategy)
    elif imputer == "simple" and imputer_strategy == "constant":
        return SimpleImputer(
            strategy=imputer_strategy, fill_value=user_input.imputer_fill_value
        )
    elif imputer == "knn":
        return KNNImputer()
    else:
        raise ValueError(
            "Invalid imputer strategy. Choose 'simple', 'knn', or 'iterative'."
        )

from typing import Dict, List, TypedDict, Union
from datetime import timedelta

from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class ScoreDict(TypedDict):
    r2_score: float
    mae: float
    mape: float


class FeatureImportances(TypedDict):
    value: float
    above_threshold: bool


# Final Dictionary Returned
class ModelInfo(TypedDict):
    scores: ScoreDict
    model: RandomizedSearchCV
    time_to_complete: timedelta


class ModelsDict(TypedDict):
    elastic_net: ModelInfo
    dtr: ModelInfo
    gbr: ModelInfo


NumericScaler = Union[StandardScaler, MinMaxScaler]

ModelGrid = List[Dict[str, BaseEstimator | str | Dict[str, List[float | int]]]]

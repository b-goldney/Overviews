from enum import Enum


class ScalerType(Enum):
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"


class ProblemType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

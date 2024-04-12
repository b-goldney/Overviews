from typing import Any, Dict, List

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    Ridge,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


REGRESSION_PARAMETER_GRID = [
    {
        "model": Lasso(),
        "name": "lasso",
        "full_name": "Lasso",
        "parameter_grid": {
            "alpha": np.logspace(-4, 1, 20),
            "max_iter": [1000, 5000, 10000],
            "tol": [1e-4, 1e-5, 1e-6],
        },
    },
    {
        "model": Ridge(),
        "name": "ridge",
        "full_name": "Ridge",
        "parameter_grid": {
            "alpha": np.logspace(-4, 1, 20),
            "max_iter": [1000, 5000, 10000],
            "tol": [1e-4, 1e-5, 1e-6],
        },
    },
    {
        "model": ElasticNet(),
        "name": "elastic_net",
        "full_name": "Elastic Net",
        "parameter_grid": {
            "alpha": np.logspace(-4, 1, 20),
            "max_iter": [1000, 5000, 10000],
            "tol": [1e-4, 1e-5, 1e-6],
        },
    },
    {
        "model": SVR(),
        "name": "svr",
        "full_name": "Support Vector Regressor",
        "parameter_grid": {
            "kernel": ["linear", "rbf", "sigmoid"],
            "C": np.logspace(-3, 3, 20),
            "epsilon": np.logspace(-3, 3, 20),
            "shrinking": [True, False],
        },
    },
    {
        "model": GradientBoostingRegressor(),
        "name": "gbr",
        "full_name": "Gradient Boosting Regressor",
        "parameter_grid": {
            "n_estimators": [50, 100, 200, 400],
            "learning_rate": [0.01, 0.1, 0.2, 0.5],
            "max_depth": [2, 3, 4, 5],
            "subsample": [0.8, 0.9, 1.0],
            "min_samples_split": [2, 3, 4],
        },
    },
    {
        "model": MLPRegressor(),
        "name": "mlpr",
        "full_name": "Multilayer Perceptron Regression",
        "parameter_grid": {
            "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 100)],
            "activation": ["relu", "tanh", "logistic"],
            "solver": ["adam", "sgd", "lbfgs"],
            "alpha": np.logspace(-4, 1, 20),
            "max_iter": [200, 500, 1000],
            "learning_rate": ["constant", "adaptive"],
        },
    },
]


CLASSIFICATION_PARAMETER_GRID = [
    {
        "model": KNeighborsClassifier(),
        "name": "knnc",
        "full_name": "K-Nearest Neighbors Classifier",
        "parameter_grid": {
            "n_neighbors": range(1, 10),
        },
    },
    {
        "model": SVC(),
        "name": "svc",
        "full_name": "Support Vector Classifier",
        "parameter_grid": {
            "kernel": ["linear", "rbf"],
            "C": [0.025, 0.5, 1, 2, 5],
            "gamma": [0.5, 1, 2, 3],
            "class_weight": ["balanced"],
        },
    },
    {
        "model": DecisionTreeClassifier(),
        "name": "dtc",
        "full_name": "Decision Tree Classifier",
        "parameter_grid": {
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "class_weight": ["balanced"],
        },
    },
    {
        "model": RandomForestClassifier(),
        "name": "rfc",
        "full_name": "Random Forest Classifier",
        "parameter_grid": {
            "max_depth": [None, 5, 10, 15, 20],
            "n_estimators": [10, 50, 100, 200],
            "max_features": ["sqrt", "log2", 1.0],
            "class_weight": ["balanced"],
        },
    },
    {
        "model": MLPClassifier(max_iter=1000),
        "name": "mlpc",
        "full_name": "Multilayer Perceptron Classifier",
        "parameter_grid": {
            "hidden_layer_sizes": [(50,), (100,), (150,)],
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1],
        },
    },
    {
        "model": AdaBoostClassifier(),
        "name": "adac",
        "full_name": "AdaBoost Classifier",
        "parameter_grid": {
            "n_estimators": [50, 100, 150, 200],
            "learning_rate": [0.001, 0.01, 0.1, 1],
        },
    },
    {
        "model": GaussianNB(),
        "name": "gnbc",
        "full_name": "Gaussian Naive Bayes",
        "parameter_grid": {},
    },
    {
        "model": QuadraticDiscriminantAnalysis(),
        "name": "qdac",
        "full_name": "Quadratic Discriminant Analysis",
        "parameter_grid": {},
    },
]

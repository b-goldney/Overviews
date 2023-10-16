import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

MODEL_GRID = [ 
    {
    "model": ElasticNet(),
    "name": "elastic_net",
    "parameter_grid": {
        "alpha": np.logspace(-4,1,20),
        "max_iter": [1000, 5000, 10000]
        } 
    },
    {
        "model": DecisionTreeRegressor(),
        "name": 'dtr',
        "parameter_grid": {
            "criterion": ["squared_error", "friedman_mse"],
            "min_samples_leaf": [2, 0.1]
        }
    },
    #{
    #    "model": SVR(),
    #    "name": 'svr',
    #    "parameter_grid": {
    #        "kernel": ["linear", "rbf", "sigmoid"],
    #        "C": np.logspace(-3,3,20),
    #        "epsilon": np.logspace(-3,3,20)
    #    }
    #},
    {
        "model": GradientBoostingRegressor(),
        "name": "gbr",
        "parameter_grid": {
            "n_estimators": [50,100,200,400],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [2,3,4,5],
            "subsample": [0.8,0.9,1.0]
        }
    }
]

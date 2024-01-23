from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import ParameterSampler
from scipy.sparse import csr_matrix, hstack

question = 'q26'
num_cols = META[question]['numeric_cols']
cat_cols = META[question]['categorical_cols']
target = META[question]['target'][0]

# cusotm scorer
def sample_weights(y_true: np.ndarray) -> np.ndarray:
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.to_numpy()
    # set up weights, using inverse weights
    threshold = 4 # sample_weights['threshold']
    less_than_threshold_count = np.sum(y_true < threshold)
    greater_or_equal_to_threshold_count = np.sum(y_true >= threshold)

    weight_less_than_threshold = 1 / less_than_threshold_count if less_than_threshold_count > 0 else 0
    weight_greater_than_threshold = 1 / greater_or_equal_to_threshold_count if greater_or_equal_to_threshold_count > 0 else 0

    importance_factor = 20 # NOTE: this can be adjusted as needed
    weight_less_than_threshold *= importance_factor
    #print(weight_less_than_threshold, weight_greater_than_threshold, '<<<< weight_less_than_threshold, ')

    # final weights
    weights = np.where(y_true < threshold, weight_less_than_threshold, weight_greater_than_threshold)
    return weights.reshape(-1)

def custom_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float: 
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.to_numpy()
    sample_weights_  = sample_weights(y_true)
    #print(np.unique(sample_weights_), '<<< sample_weights', flush=True)
    return  mean_squared_error(y_true, y_pred, sample_weight=sample_weights_)

make_scorer(custom_scorer, greater_is_better=False)
#weighted_scorer = make_scorer(custom_weighted_scorer, greater_is_better=False)

# create RanomizedSearchCV instance
#reg = DecisionTreeRegressor(random_state=0, min_samples_split=0.02)
dt_param_grid = {
    'max_depth': [ 10, 20, 30],
    'min_samples_split': [0.02, 0.05, 0.10],
    'min_samples_leaf': [0.01, 0.02, 0.05]
}

gb_param_grid =  {
    # gradient boosting regressor
"n_estimators": [50,100,200,400],
"learning_rate": [0.01, 0.1, 0.2],
"max_depth": [2,3,4,5],
"subsample": [0.8,0.9,1.0]
}

#param_grid = list(ParameterSampler(param_grid, n_iter=6))
#print(param_grid)

# data clean up
df2 = df.copy()
#df2 = df2[ ~df2[target].isnull() ].reset_index(drop=True)
df2 = df2.dropna(subset=target)
df2 = df2[df2[target] != 0].reset_index(drop=True)
df2 = df2[num_cols + cat_cols + [ target ]]
df2 = df2.fillna(0)

# train test split
x_train, x_test, y_train, y_test = train_test_split(
    df2[num_cols + cat_cols], df2[[ target ]], test_size=0.20)

# standardize data
column_transformer = ColumnTransformer(
    transformers = [
        ("standard_scaler", StandardScaler(), num_cols),
        ("categorical_encoder", OneHotEncoder(), cat_cols),
    ]
)
x_train = column_transformer.fit_transform(x_train)
print(x_train.shape)

# calculate weights
frequency = Counter(y_train[target].values)
weights = sample_weights(y_train)
weights = np.array([1 / frequency[x] for x in y_train[target]])

best_score = np.inf
random_search = RandomizedSearchCV(GradientBoostingRegressor(), 
                                   gb_param_grid, 
                                   n_iter=10, 
                                   refit=True,
                                   scoring=make_scorer(custom_scorer, greater_is_better=False))
random_search.fit(x_train, y_train[target].ravel())
y_predict = pd.DataFrame( random_search.predict(x_train), columns=['y_predict'] )

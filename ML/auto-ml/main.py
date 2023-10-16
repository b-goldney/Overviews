import pandas as pd

import AutoMl from .auto_ml

# load and clean data
df = pd.read_csv('..//california-housing.csv', header=0)
df.columns = [x.lower() for x in df.columns]
df.columns = [ x.lower().replace(' ', '_') for x in df.columns ]
print(df.head(2))
print(df.shape)
    
# fit model
auto_ml = AutoMl(df, 
                num_cols=all_num_cols, 
                cat_cols=all_cat_cols, 
                target=target, #META['q22']['target'], 
                numeric_scaler=StandardScaler, 
                categorical_encoder=OneHotEncoder,
                test_size=0.25
                )

num_cols = [ 'latitude', 'longitude', 'housing_median_age', 'total_rooms' ,  
            'total_bedrooms', 'population', 'households', 'median_income'] 
cat_cols = ['ocean_proximity']
target = ['median_house_value']

# fit model
auto_ml = AutoMl(df, 
                num_cols=num_cols, 
                cat_cols=cat_cols, 
                target=target,  
                numeric_scaler=StandardScaler, 
                categorical_encoder=OneHotEncoder,
                test_size=0.75
                )
auto_ml.preprocesing()
all_models = auto_ml.run_models()
predictions = auto_ml.get_predictions

# output
auto_ml.get_predictions['original_data'] 
auto_ml.get_predictions['processed_data']
print( auto_ml.feature_importance_values )

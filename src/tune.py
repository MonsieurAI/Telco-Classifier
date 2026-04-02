import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from preprocess import load_train, load_preprocessor
import xgboost as xgb
from scipy.stats import uniform
import json

# Get data
X, y = load_train()
preprocessor = load_preprocessor(X)

# Build the pipeline
xgb_model = xgb.XGBClassifier(device='cuda')
pipe = Pipeline([('preprocess',preprocessor), ('model', xgb_model)])

# Tuning
param_dict = {
    'model__max_depth':[3,5,7,9],
    'model__min_child_weight':[1,3,5,7],
    'model__learning_rate':uniform(0.01,0.2),
    'model__gamma':uniform(0,0.3),
    'model__subsample':uniform(0.6,0.4)
}
rs = RandomizedSearchCV(pipe,param_dict,n_iter=50,scoring='f1',cv=5,random_state=67,verbose=1)

rs.fit(X, y)

# Print results
results=pd.DataFrame(rs.cv_results_)
best_params = rs.best_params_
best_params = {key.split('__')[1] : value for key, value in best_params.items()}
print(f'Best score: {rs.best_score_}')
print('Parameters:')
print(best_params)

# Save all and the best results
results.to_csv('../data/tuning_results.csv')
with open('../data/tuning_best.json', 'w') as file:
    json.dump(best_params, file)
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from preprocess import load_and_clean_data,load_preprocessor
import json

X,y = load_and_clean_data()
preprocessor = load_preprocessor(X)

with open('../data/tuning_best.json', 'r') as file:
    params = json.load(file)

model=xgb.XGBClassifier(**params)
pipe = Pipeline([('preprocessing',preprocessor),('model_xgb',model)])

print(cross_val_score(pipe,X,y,scoring='f1'))
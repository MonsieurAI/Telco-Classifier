import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from preprocess import load_and_clean_data,load_preprocessor

X,y = load_and_clean_data()
preprocessor = load_preprocessor(X)

model=xgb.XGBClassifier()
pipe = Pipeline([('preprocessing',preprocessor),('model_xgb',model)])

print(cross_val_score(pipe,X,y,scoring='f1'))
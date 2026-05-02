import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from preprocess import load_train, load_test, load_preprocessor
import json
import joblib
from paths import TUNING_BEST_PATH,FINAL_PIPELINE_PATH

X_train,y_train = load_train()
X_test,y_test = load_test()
preprocessor = load_preprocessor(X_train)

with open(TUNING_BEST_PATH, 'r') as file:
    params = json.load(file)

model=xgb.XGBClassifier(**params)
pipe = Pipeline([('preprocessing',preprocessor),('model_xgb',model)])
pipe.fit(X_train, y_train)

predictions = pipe.predict(X_test)
final_f1 = f1_score(y_test, predictions)

print(f'Final f1 score: {final_f1}')

# Export the pipeline
joblib.dump(pipe, FINAL_PIPELINE_PATH)
import pandas as pd
from sklearn.model_selection import train_test_split
from paths import RAW_DATA_PATH,RAW_TRAIN_PATH,RAW_TEST_PATH

data = pd.read_csv(RAW_DATA_PATH)
train, test = train_test_split(data, test_size=0.15, stratify = data['Churn'], random_state=42)

train.to_csv(RAW_TRAIN_PATH,index=False)
test.to_csv(RAW_TEST_PATH,index=False)
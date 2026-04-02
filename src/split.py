import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('../data/raw-data.csv')
train, test = train_test_split(data, test_size=0.15, stratify = data['Churn'], random_state=42)

train.to_csv('../data/raw-train.csv',index=False)
test.to_csv('../data/raw-test.csv',index=False)
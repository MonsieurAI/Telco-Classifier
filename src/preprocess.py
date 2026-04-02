import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder

def clean(data):
    data = data.drop(['customerID'], axis=1)
    data['SeniorCitizen'] = data['SeniorCitizen'].apply(lambda x: 'Yes' if x == 1 else 'No')
    data = data[data['TotalCharges'] != ' ']
    data['TotalCharges'] = data['TotalCharges'].astype(float)
    data = data.reset_index(drop=True)

    X = data.drop('Churn', axis=1)
    y = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    return X,y

def load_train():
    train = pd.read_csv('../data/raw-train.csv')
    X,y = clean(train)
    return X,y

def load_test():
    test = pd.read_csv('../data/raw-test.csv')
    X,y = clean(test)
    return X,y

def load_preprocessor(X):
    num_cols = X.select_dtypes(include='number').columns
    cat_cols = X.select_dtypes(include='object').columns
    encoder=OneHotEncoder(drop='first',sparse_output=False)
    normalizer=StandardScaler()
    preprocessor=ColumnTransformer([('enc',encoder,cat_cols),('norm',normalizer,num_cols)],remainder='passthrough')
    return preprocessor
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder

def clean(data):
    # Drop id column
    # Drop statistically unnecessary "gender" and "PhoneService" columns
    data = data.drop(['customerID','gender','PhoneService'], axis=1)

    data['SeniorCitizen'] = data['SeniorCitizen'].map({1:'Yes', 0:'No'})

    # Replace empty "TotalCharges" values with "MonthlyCharges" * "tenure"
    calculated_total_charges = data['MonthlyCharges'] * data['tenure']
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'],errors='coerce').fillna(calculated_total_charges)

    data = data.reset_index(drop=True)

    X = data.drop('Churn', axis=1)
    y = data['Churn'].map({'Yes':1, 'No':0})
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
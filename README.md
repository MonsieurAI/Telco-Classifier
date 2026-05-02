## Overview
This repository contains an end-to-end, production-ready machine learning pipeline designed to predict customer churn using the Telco dataset.

The system utilizes an XGBoost classifier integrated with Scikit-Learn’s ColumnTransformer to ensure prevention of data leakage during training and inference.

## Mathematical Justifications & Preprocessing
Unlike standard implementations that rely on blind imputation or arbitrary feature selection, the data cleaning architecture in preprocess.py is entirely deterministic:

* Categorical Independence (Chi-Squared): Features such as gender and PhoneService were dropped from the pipeline after statistical testing proved they lacked a significant relationship with the target variable.
* Deterministic Imputation: To guarantee production stability and prevent pipeline crashes from NaN values or empty string artifacts (' '), TotalCharges is deterministically imputed using the calculated product of MonthlyCharges and tenure.

## Model Architecture & Performance
The predictive engine is an XGBoost model optimized via hyperparameter tuning (stored in tuning_best.json).

#### Final F1 Score: ~0.58

## System Deployment
The trained pipeline is serialized as a binary artifact (final_pipeline.joblib).

## How to run locally
Clone the repository and move to the directory

```bash
git clone https://github.com/MonsieurAI/Telco-Classifier.git
cd Telco-Classifier
```

Create and activate virtual environment

```bash
# On Windows:
python -m venv venv
venv/Scripts/activate

# On Mac/Linux:
python3 -m venv venv
source venv/bin/activate
```

Install required dependencies

```bash
pip install -r requirements.txt
```

To reproduce the model from the raw CSV files, execute the training script. This will output the F1 metrics to the terminal and generate a production_pipeline.joblib file in the models/ directory

```bash
python src/train.py
```
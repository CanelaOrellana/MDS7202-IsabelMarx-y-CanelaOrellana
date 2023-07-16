"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.11
"""

import logging

import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


from typing import Dict

import pandas as pd


def split_data(data: pd.DataFrame, params: dict):

    shuffled_data = data.sample(frac=1, random_state=params["random_state"])
    rows = shuffled_data.shape[0]

    train_ratio = params["train_ratio"]
    valid_ratio = params["valid_ratio"]

    train_idx = int(rows * train_ratio)
    valid_idx = train_idx + int(rows * valid_ratio)

    assert rows > valid_idx, "test split should not be empty"

    target = params["target"]
    X = shuffled_data.drop(columns=target)
    y = shuffled_data[[target]]

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_valid, y_valid = X[train_idx:valid_idx], y[train_idx:valid_idx]
    X_test, y_test = X[valid_idx:], y[valid_idx:]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_mae")["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")

    return best_model


# TODO: completar train_model
def train_model(X_train,y_train,X_valid,y_valid):
    LR = LinearRegression()
    RF = RandomForestRegressor()
    SVM = SVR()
    XGB = XGBRegressor()
    LGBMR = LGBMRegressor()

    id_train = mlflow.create_experiment("experimento_train")
    mlflow.autolog() # registrar automáticamente información del entrenamiento
    #Linear Regression
    with mlflow.start_run(run_name="run LR", experiment_id=id_train): # delimita inicio y fin del run
        # aquí comienza el run
        LR.fit(X_train, y_train) # train the model
        y_pred = LR.predict(X_valid) # Use the model to make predictions on the test dataset.
        valid_mae = mean_absolute_error(y_valid,y_pred)
        mlflow.log_metric("valid_mae", valid_mae)
        # aquí termina el run
    #Random Forest
    with mlflow.start_run(run_name="run RF", experiment_id=id_train): # delimita inicio y fin del run
        # aquí comienza el run
        RF.fit(X_train, y_train) # train the model
        y_pred = RF.predict(X_valid) # Use the model to make predictions on the test dataset.
        valid_mae = mean_absolute_error(y_valid,y_pred)
        mlflow.log_metric("valid_mae", valid_mae)
        # aquí termina el run
    #Support Vector Regression
    with mlflow.start_run(run_name="run SVR", experiment_id=id_train): # delimita inicio y fin del run
        # aquí comienza el run
        SVM.fit(X_train, y_train) # train the model
        y_pred = SVM.predict(X_valid) # Use the model to make predictions on the test dataset.
        valid_mae = mean_absolute_error(y_valid,y_pred)
        mlflow.log_metric("valid_mae", valid_mae)
        # aquí termina el run
    #XGBoost Regressor
    with mlflow.start_run(run_name="run XGB", experiment_id=id_train): # delimita inicio y fin del run
        # aquí comienza el run
        XGB.fit(X_train, y_train) # train the model
        y_pred = XGB.predict(X_valid) # Use the model to make predictions on the test dataset.
        valid_mae = mean_absolute_error(y_valid,y_pred)
        mlflow.log_metric("valid_mae", valid_mae)
        # aquí termina el run
    #Light GBM Regressor
    with mlflow.start_run(run_name="run LGBMR", experiment_id=id_train): # delimita inicio y fin del run
        # aquí comienza el run
        LGBMR.fit(X_train, y_train) # train the model
        y_pred = LGBMR.predict(X_valid) # Use the model to make predictions on the test dataset.
        valid_mae = mean_absolute_error(y_valid,y_pred)
        mlflow.log_metric("valid_mae", valid_mae)
        # aquí termina el run

    best_model = get_best_model(id_train)
    return best_model
    


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info(f"Model has a Mean Absolute Error of {mae} on test data.")

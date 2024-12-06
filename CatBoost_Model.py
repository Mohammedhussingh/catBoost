import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import folium
from folium.plugins import HeatMap
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoost, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from joblib import dump, load

"""
This script defines two classes, `Data_Prep` and `Model`, to preprocess real estate data and train a predictive model 
using CatBoostRegressor. The goal is to predict property prices based on the features provided in the dataset.

Key Features:
-------------
1. **Data Preparation (Data_Prep Class)**:
   - Reads the dataset from a given file link.
   - Prepares the feature matrix (`X`) and target variable (`y`).
   - Normalizes the data using Min-Max scaling.
   - Splits the data into training and test sets.

2. **Model Training and Evaluation (Model Class)**:
   - Initializes a CatBoostRegressor model with specified hyperparameters.
   - Trains the model on the normalized data with a log-transformed target variable.
   - Saves the trained model to a file for future use.
   - Evaluates the model on the test set using metrics such as RMSE and R².

Classes:
--------
1. **Data_Prep**:
    Handles data preprocessing, including reading, normalization, and train-test splitting.

    Methods:
    --------
    __init__(Link: str):
        Initializes the class with the file path to the dataset and prepares feature and target variables.

    read_data() -> pd.DataFrame:
        Reads the dataset from the specified file link.

    Normalize_Data() -> np.ndarray:
        Normalizes the feature data using Min-Max scaling.

    Spliter() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Splits the normalized data into training and test sets and log-transforms the target variable.

2. **Model**:
    Trains and evaluates the CatBoostRegressor model.

    Methods:
    --------
    __init__(link1: str):
        Initializes the class with a dataset file path and prepares the training and test data using `Data_Prep`.

    fit() -> CatBoostRegressor:
        Trains the CatBoost model on the training data and saves it as a file.

    predict():
        Evaluates the model on the test set and prints the RMSE and R² scores.

Usage:
------
1. Prepare the dataset and save it to a file.
2. Initialize the `Model` class with the file path to the dataset.
3. Call the `fit()` method to train the model.
4. Call the `predict()` method to evaluate the model and view metrics.

Example:
--------
from data_model import Data_Prep, Model

# File path to the dataset
Data_link = "/path/to/your/dataset.csv"

# Initialize and train the model
model = Model(Data_link)
model.fit()
model.predict()

Dependencies:
-------------
- pandas
- numpy
- sklearn (MinMaxScaler, train_test_split, mean_squared_error, r2_score)
- catboost (CatBoostRegressor)
- joblib (for saving the model)
"""


class Data_Prep:
    def __init__(self, Link) -> None:
        self.Link = Link
        self.Data = self.read_data()

        self.X = self.Data.drop(
            columns=["Price", "Id"]  # ,'sqdm_price','Locality_mean_Price'
        )  # All columns except 'Price' are features
        self.y = self.Data["Price"]  # Target variable is 'Price'

        pass

    def read_data(self):
        Data = pd.read_csv(self.Link, index_col=0)

        return Data

    def Normalize_Data(self):
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(self.X)
        return normalized_data

    def Spliter(self):
        X_ = self.Normalize_Data()
        y1 = self.y

        X_train, X_test, y_train, y_test = train_test_split(
            X_, y1, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test


from catboost import CatBoostRegressor
import numpy as np
from joblib import dump
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class Model1:
    def __init__(self, link1) -> None:
        # Initialize CatBoostRegressor
        self.link = link1
        self.model = CatBoostRegressor(iterations=3000,  # Number of boosting iterations
                          learning_rate=0.01,  # Learning rate
                          depth=5,  # Depth of the trees
                          random_seed=42  # Random seed for reproducibility  
                          ,eval_metric="RMSE",
                          l2_leaf_reg=10,
                          min_data_in_leaf=10
                          )
        Data_obj = Data_Prep(self.link)
        self.X_train, self.X_test, self.y_train, self.y_test = Data_obj.Spliter()

    def fit(self):
        # Log-transform the target variable
        y_train_log = np.log1p(self.y_train)

        self.model.fit(self.X_train, y_train_log)
        # Save the model to a file
        dump(self.model, "model_Hussain.joblib")
        return self.model

    def evaluate_metrics(self, y_true, y_pred, dataset_name="Test"):
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        smape = np.mean(
            2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))
        ) * 100

        # Print results
        print(f"{dataset_name} Metrics:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  sMAPE: {smape:.2f}%")
        print()
        return mae, rmse, r2, mape, smape

    def predict_(self):
        # Predict directly on the original scale
        y_train_pred = np.expm1(self.model.predict(self.X_train))
        y_test_pred = np.expm1(self.model.predict(self.X_test))

        print("Evaluating Metrics on Original Scale:")
        self.evaluate_metrics(self.y_train, y_train_pred, dataset_name="Training")
        self.evaluate_metrics(self.y_test, y_test_pred, dataset_name="Test")
        return
Data_link = "ED.csv"

Model = Model1(Data_link)


Model.fit()

Model.predict_()

from Data_Prepration import Data_cleaning ,Feature_Engineering
from CatBoost_Model import Data_Prep,Model1

'''
Main Script for Real Estate Price Prediction

This script orchestrates the data preparation, feature engineering, and model training process 
for predicting real estate prices. It uses modules for data cleaning, feature engineering, 
and machine learning modeling with CatBoost.

Modules:
--------
1. **Data_Preparation**:
   - Contains the `Data_cleaning` and `Feature_Engineering` classes for preprocessing and feature engineering.
   - Responsible for handling raw data and adding additional engineered features.

2. **CatBoost_Model**:
   - Contains the `Data_Prep` and `Model1` classes for data preparation and model training using CatBoostRegressor.

Workflow:
---------
1. **Data Cleaning**:
   - The raw dataset is processed using the `Data_cleaning` class to encode categorical features 
     and discard unnecessary columns.
   - The cleaned dataset is saved as `Data_Engineering.csv`.

2. **Feature Engineering**:
   - Additional features such as coastal locality indicators, GDP, average rent prices, 
     and bedrooms per area are added to the dataset.
   - The feature-engineered dataset is saved as `ED.csv`.

3. **Model Training**:
   - The processed dataset is passed to the `Model1` class for normalization, splitting, 
     training, and evaluation using CatBoostRegressor.
   - The model is evaluated using RMSE and R² metrics.

Dependencies:
-------------
- pandas
- numpy
- scikit-learn (for data splitting and metrics)
- catboost (for modeling)
- MinMaxScaler (for normalization)

Files:
------
1. `Final_cleaned_Data.csv` - Raw data file for preprocessing.
2. `Data_Engineering.csv` - Intermediate file after cleaning.
3. `ED.csv` - Final dataset after feature engineering.

Usage:
------
1. Ensure all required modules (`Data_Prepration` and `CatBoost_Model`) are correctly imported.
2. Update the `link` and `Data_link` variables with the correct file paths.
3. Run the script to preprocess the data, engineer features, and train the model.

Output:
-------
1. Trained CatBoost model saved to disk.
2. Printed RMSE and R² scores for model evaluation.'''

link ="/home/learner/Desktop/ImmoLiza_reg/Regression_Hussain/Data/Final_cleaned_Data.csv"
Program=Data_cleaning(link)
Encoded=Program.save1()

DE=Feature_Engineering(Encoded)
DE.Save()

Data_link="/home/learner/Desktop/ImmoLiza_reg/Regression_Hussain/Python Files/ED.csv"

X = Model1(Data_link)  # Instantiate the object
X.fit()               # Call methods on the object
X.predict_()           # Call the predict method



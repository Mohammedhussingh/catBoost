# ImmoEliza Real Estate Price Prediction

## Project Description

Welcome to the **ImmoEliza Real Estate Price Prediction** repository using catBoost!  
This project focuses on predicting real estate prices in Belgium using machine learning techniques. We preprocess the dataset, engineer meaningful features, train a predictive model, and evaluate its performance. The objective is to deliver a robust and accurate model for real estate price prediction while following best coding practices.

---

## Installation

### Prerequisites

Ensure the following are installed:
- Python 3.8+
- `pip` package manager

### Clone the Repository
```bash
git clone https:[//github.com//challenge-regression.git](https://github.com/Mohammedhussingh/catBoost)

```
### Files:
#### Python Files
1. **`main.py`**: Runs the whole program.  
2. **`Catboost_Model.py`**: Contains the model training and prediction logic, built with hyperparameter tuning.  
3. **`Data_preparation.py`**: Handles data cleaning and feature engineering.

   
#### Data Files
1. `Final_cleaned_Data.csv` - Raw data file for preprocessing.
2. `Data_Engineering_pre.csv` - Intermediate file after cleaning.
3. `ED.csv` - Final dataset after feature engineering.


### Usage

### Run the Script
Execute the model training pipeline with the following command:
```bash
python main.py
```

### Personal Insights
- **Biggest Challenge**: Identifying and engineering impactful features.
- **Key Learning**: Feature selection significantly affects model accuracy.

### Evaluation Metrics

- **Root Mean Squared Error (RMSE)**: Measures prediction error magnitude.
- **R² Score**: Indicates the proportion of variance explained by the model.
- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in predictions, without considering their direction.
- **RMSE (on training/test)**: Measures the root mean squared error for both the training and test datasets.
- **R² (on training/test)**: Indicates the proportion of variance explained by the model on both the training and test datasets.
- **Mean Absolute Percentage Error (MAPE)**: Measures the accuracy of predictions as a percentage, providing insight into relative error.
- **Symmetric Mean Absolute Percentage Error (sMAPE)**: Similar to MAPE but reduces bias in the percentage error for extreme values in predictions..

## Challenges and Limitations

- **Limited Data**: The dataset contains approximately 10,000 records. While this allows the model to achieve an acceptable R² and good RMSE, the accuracy and performance are expected to improve with a larger dataset.
- **Scalability**: As the dataset expands, additional preprocessing, feature engineering, and model adjustments may be necessary to maintain or improve the model’s performance.
- **Hyperparameter Tuning and Cross-Validation**: Several trials were performed, including hyperparameter tuning and cross-validation. These techniques helped improve model performance, but the results are still limited by the size of the dataset. More data would likely allow for better fine-tuning and generalization.

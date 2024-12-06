# Project Overview: Predicting Property Prices Using CatBoost
### Introduction:
This project focuses on developing a predictive model for real estate property prices using machine learning techniques. Leveraging a dataset of 10,684 property records, the objective is to accurately estimate property prices based on key features such as size, location, condition, and other relevant attributes.

The CatBoostRegressor, known for its efficiency and ability to handle complex datasets, was chosen as the core model. The project involves thorough data preprocessing, feature engineering, and hyperparameter tuning to ensure robust model performance.

The end goal is to provide actionable insights into property valuation, helping stakeholders like real estate professionals, urban planners, and investors make data-driven decisions. By optimizing the prediction process, the model aims to deliver high accuracy while maintaining interpretability and scalability.

### Data Features:
A total of 10,684 records were cleaned of outliers and missing values. The dataset contains 18 features, including 5 engineered ones:

1. **Price (Target)**: Represents the property cost.
2. **Bedrooms**: Number of bedrooms.
3. **Living_Area**: Living space size (e.g., square meters).
4. **Is_Equiped_Kitchen**: Binary feature indicating an equipped kitchen.
5. **Terrace**: Binary feature for terrace presence.
6. **Garden**: Binary feature for garden presence.
7. **State**: Property condition (new, refurbished, old).
8. **Facades**: Number of building facades.
9. **Locality_encoded**: Encoded representation of property locality.
10. **Type_encoded**: Encoded property type (apartment, house).
11. **SubType_encoded**: Encoded property subtype (e.g., penthouse).
12. **Prov_encoded**: Encoded province.
13. **Region_encoded**: Encoded geographical area (e.g., urban vs. rural).
14. **Is_On_Coast** (engineered): Binary feature indicating coastal location.
15. **GDP** (engineered): Economic health of the province.
16. **Avg_rent** (engineered): Average rental price in the area.
17. **Avg_price** (engineered): Average property price in the region.
18. **Bedrooms_per_area** (engineered): Ratio of bedrooms to living area.

### The Model:
The model is saved as `model_Hussain.joblib`. Parameters were chosen based on hyperparameter tuning (details in `CatBoost_Building.ipynb`):

- **iterations=3000**: Number of boosting iterations.
- **learning_rate=0.01**: Step size per iteration.
- **depth=5**: Tree depth for complexity balance.
- **random_seed=42**: Ensures reproducibility.
- **eval_metric="RMSE"**: Metric to optimize regression accuracy.
- **l2_leaf_reg=10**: Regularization to prevent overfitting.
- **min_data_in_leaf=10**: Minimum samples in each leaf to avoid noise-fitting.

### Feature Importances:
Feature importance analysis highlights:
1. **Living_Area**: Most significant, mean SHAP value 0.19.
2. **State**: SHAP value 0.07.
3. **SubType_encoded**, **Bedrooms**, **Avg_rent**, and **Avg_price**: SHAP value 0.05.
4. **Locality_encoded**, **Facades**, **Region_encoded**, and **GDP**: Moderate importance.
5. **Type_encoded**, **Is_Equiped_Kitchen**, and others: Minimal impact.

### Visual Representations:
- **Normalized SHAP Feature Importances**: Highlights the significance of individual features.
- **Bar Charts**: Show detailed importance rankings.

### Model Performance:

### Metrics on Test Data:
- **MAE**: 71,859.98
- **RMSE**: 100,826.30
- **R²**: 0.7134
- **MAPE**: 21.34%
- **sMAPE**: 19.39%

### Metrics on Training Data:
- **MAE**: 65,873.13
- **RMSE**: 91,633.47
- **R²**: 0.7429
- **MAPE**: 20.05%
- **sMAPE**: 18.28%

### Explanation:
The test data's **R²** of 0.7134 indicates a moderate fit, while the **MAE** and **RMSE** suggest room for improvement. Training data shows similar performance (**R²**: 0.7429), but the slightly higher **MAPE** hints at overfitting. Attempts to add more features led to overfitting due to dataset size, so these features were removed.

## CatBoost Model Performance:

- **Training Performance**: 4.3 seconds (all data are numerical).
- **Cross-validation** (for all model parameters, at least 4 values per parameter) & **Hyperparameter Tuning**: 3-7 minutes.
- **Prediction**: Instant.

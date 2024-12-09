# Project Overview: Predicting Property Prices Using CatBoost
### Introduction:
This project focuses on developing a predictive model for real estate property prices using machine learning techniques. Leveraging a dataset of 10,684 property records, the objective is to accurately estimate property prices based on key features such as size, location, condition, and other relevant attributes.

The CatBoostRegressor, known for its efficiency and ability to handle complex datasets, was chosen as the core model. The project involves thorough data preprocessing, feature engineering, and hyperparameter tuning to ensure robust model performance.

The end goal is to provide actionable insights into property valuation, helping stakeholders like real estate professionals, urban planners, and investors make data-driven decisions. By optimizing the prediction process, the model aims to deliver high accuracy while maintaining interpretability and scalability.

### Data Features:

A total of 10,684 records were cleaned of outliers and missing values. The dataset contains 18 features, including 5 engineered ones:

1. **Price (Target)**: The dependent variable you're trying to predict, representing the cost of a property.

2. **Bedrooms**: The number of bedrooms in the property. More bedrooms typically correlate with higher property prices.

3. **Living_Area**: The size of the living space in the property (e.g., square meters). Larger living areas generally increase property price.

4. **Is_Equiped_Kitchen**: A binary feature indicating whether the property has an equipped kitchen. Properties with an equipped kitchen can be valued higher.

5. **Terrace**: A binary feature indicating if the property has a terrace. This can add value, depending on the property location.

6. **Garden**: A binary feature indicating the presence of a garden. Similar to terrace, gardens can increase a property’s value.

7. **State**: Refers to the condition of the property (e.g., new, refurbished, or old). The state of the property influences its market price.

8. **Facades**: The number of building facades or exterior walls. More facades could indicate a larger or more aesthetically desirable property.

9. **Locality_encoded**: A numerical encoding of the property’s locality, which helps represent the location's impact on the price.

10. **Type_encoded**: Encodes the property type (e.g., apartment, house), which can significantly affect its price.

11. **SubType_encoded**: Encodes the specific subtype of the property (e.g., duplex, penthouse), which also influences pricing.

12. **Prov_encoded**: Encodes the province or region within a country where the property is located, which may influence the price due to local economic conditions.

13. **Region_encoded**: Similar to the province but might refer to a larger geographical area (e.g., urban vs rural).

14. **Is_On_Coast**(engineerd):: A binary feature indicating whether the property is located on the coast. Coastal properties often have higher prices due to location desirability.

15. **GDP**(engineerd):: The Gross Domestic Product of the province, reflecting the overall economic health of the place, which could influence property prices.

16. **Avg_rent**(engineerd): The average rental price in the area. Higher average rents typically indicate higher property values .

17. **Avg price**(engineerd):: The average price of properties in the locality or region. It gives a reference for pricing trends (obtained from an external source, not from the data itself).

18. **Bedrooms_per_area**(engineerd): The ratio of the number of bedrooms to the living area, which can affect property price by indicating space efficiency.


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
   


![normalized_shap_importance_with_labels](https://github.com/user-attachments/assets/a9ba3c77-afbb-4be2-91cb-291d2ff3b3cc)

![e325bc1a-fb74-4e9a-aba7-6a46b550ad6b](https://github.com/user-attachments/assets/90b6a3e8-221b-433f-b017-299c53033d7d)

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
The test data's **R²** of 0.7134 indicates a moderate fit, while the **MAE** and **RMSE** suggest room for improvement. Training data shows similar performance (**R²**: 0.7429) as Test data (**R²**: 0.7134) so mosr probably we dont have overfiting here.Attempts to add more features led to overfitting due to dataset size, so these features were removed.

## CatBoost Model Performance:

- **Training Performance**: 4.3 seconds (all data are numerical).
- **Cross-validation** (for all model parameters, at least 4 values per parameter) & **Hyperparameter Tuning**: 3-7 minutes.
- **Prediction**: Instant.

## Note:
While this model (CatBoost) is suitable and performs well, the scale and quality of the data or potentially the techniques used for data collection—limit its effectiveness. This is a key weakness of the model. With access to better-quality data and wider scale, significantly improved performance could be achieved.

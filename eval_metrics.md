# Introduction:
The CatBoostRegressor is optimized for high performance and can handle categorical data well, but in your case, it is being used with numerical data (since it’s a regression task and based on the requirements of the project). The used data is calculated from the data-extracting features phase.

# Data Features:
A total of 10,684 data records have been cleaned of outliers, and there are no missing or non-values, with 18 features( 5 engineerd):

1. **Price(Target)**: The dependent variable you're trying to predict, representing the cost of a property.

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



# The Model:
My modeld saved in model_Hussain.joblib  with parametrs chosen based on hyperparameter tuining (you can find it in CatBoost_Buliding.ipynb ) 


- **iterations=3000**: 
  - Defines the number of boosting iterations (trees). 

- **learning_rate=0.01**: 
  - Controls the step size for each iteration.

- **depth=5**: 
  - Controls the complexity of the decision trees. A depth of 5 strikes a balance, capturing enough complexity while avoiding overfitting.

- **random_seed=42**: 
  - Ensures reproducibility of the results. A fixed random seed allows the same outcomes across multiple runs of the model.

- **eval_metric="RMSE"**: 
  - The evaluation metric for the model. RMSE penalizes larger errors, making it useful for regression tasks to optimize the prediction accuracy.

- **l2_leaf_reg=10**: 
  - Regularization term to prevent overfitting. 

- **min_data_in_leaf=10**: 
  - Defines the minimum number of samples in each leaf. Ensures that the model does not overfit to noise by requiring a reasonable number of data points per leaf.


### Feature Importances

The following bar chart displays the importance of different features in predicting the target variable (Price). Features like **Living_Area**, **Locality_encoded**, and **State** have the highest importance, while **Garden** and **Is_Equiped_Kitchen** have lower importance.


## SHAP Feature Importances

The following features have the most significant impact on the model's output, based on their mean SHAP values:

1. **Living_Area**: This feature has the most significant impact on the model's output, with a mean SHAP value of 0.19.
2. **State**: The second most influential feature, with a SHAP value of 0.07.
3. **SubType_encoded**: Contributes a SHAP value of 0.05.
4. **Bedrooms**: Also contributing 0.05, alongside other features like **Avg_rent** and **Avg_price**.

Other notable features include:
- **Locality_encoded**: 0.05
- **Facades**: 0.04
- **Region_encoded**: 0.02
- **Terrace**: 0.02
- **GDP**: 0.02
- **Bedrooms_per_area**: 0.01
- **Type_encoded**, **Prov_encoded**, **Is_Equiped_Kitchen**, **Garden**, and **Is_On_Coast** have a relatively minimal impact.

### Visual Representation
![normalized_shap_importance_with_labels](https://github.com/user-attachments/assets/17c7ea87-8ef6-4bff-b74e-1527cc9c608e)


![e325bc1a-fb74-4e9a-aba7-6a46b550ad6b](https://github.com/user-attachments/assets/796485c8-d62d-46e2-ab01-e15047591959)



### The model performance:

## Model Performance Metrics

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
The performance on the test data shows a moderate fit, with an **R²** of 0.7134, indicating that around 71.34% of the variance in the test set is explained by the model. However, the **MAE** and **RMSE** are still relatively high, suggesting potential areas for improvement in model accuracy.

For the training data, the model's performance is somewhat similar, with an **R²** of 0.7429, which reflects that the model explains about 74.29% of the variance in the training data. The **MAE** and **RMSE** are lower compared to the test data, showing that the model is more closely fitted to the training set. The slight increase in **MAPE** and **sMAPE** indicates that the model is likely experiencing some degree of overfitting.

Given that the data was split into 80% for training and 20% for testing, the discrepancies between the two sets suggest that further tuning is needed to reduce overfitting and improve generalization to unseen data. I experimented with adding more features to improve the model’s performance, but this approach led to overfitting due to the relatively small size of the dataset. As a result, I removed the additional features to prevent the model from being too tightly fitted to the training data, which could compromise its ability to generalize well on new data.

# CatBoost Model Performance

- **Training Performance**: 4.3 seconds (since all data are numerical).
- **Cross-validation (for all model parameters with at least 4 for each) & Hyperparameter Tuning**: Takes additional time during these processes (3-7 minutes).
- **Prediction**: Instant.


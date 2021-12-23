# Housing Price Prediction

## Pipeline
- EDA
- Data preprocessing: handling missing values and feature engineering
- Model fitting (benchmark: linear regression)
- Hyperparameter tuning
- Model evaluation

## Results:
MSE scores:
- Lasso:  158.16
- Ridge:  157.74
- Elastic Net: 158.68
- Random Forest Regressor: 87.25
- Light GBM: 79.23
- XGBoost Regressor: 75.34

Based on above evaluation, it is clear that XGBoost Regressor performs best. It also has the lowest cross-validation MSE score.

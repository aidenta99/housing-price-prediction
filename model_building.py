from data_cleaning import final_features
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder

target_column = "SalePrice"

# Split data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    final_features.drop(columns=target_column), final_features[target_column]
)

# Log transform Sale Price
y_train = pd.DataFrame(np.log(y_train)).reset_index(drop=True)

from sklearn.model_selection import cross_val_score
# Baseline model: Linear Regression
lm = LinearRegression()
lm.fit(X_train, y_train)
np.mean(cross_val_score(lm, X_train, y_train, scoring = 'neg_mean_absolute_error'))

# Lasso regression
lm_l = Lasso()
lm_l.fit(X_train, y_train)
np.mean(cross_val_score(lm_l, X_train, y_train, scoring = 'neg_mean_absolute_error'))

# Hyper-parameter tuning for Lasso Regression
alpha_l = []
error_l = []

for i in range(1,100):
    alpha_l.append(i/10000)
    lml = Lasso(alpha=(i/10000))
    error_l.append(np.mean(cross_val_score(lml, X_train, y_train, scoring = 'neg_mean_absolute_error')))
    
plt.plot(alpha_l,error_l)
plt.savefig('img/lasso_alpha-vs-error.png')

lasso_parameters = {'alpha':[i/10000 for i in range(1,100)]}
gs_lasso = GridSearchCV(lm_l, lasso_parameters, scoring='neg_mean_absolute_error')
gs_lasso.fit(X_train, y_train)
gs_lasso.best_score_
gs_lasso.best_estimator_

# Ridge Regression
lm_r = Ridge()
lm_r.fit(X_train, y_train)
np.mean(cross_val_score(lm_r, X_train, y_train, scoring = 'neg_mean_absolute_error'))

# Hyper-parameter tuning for Ridge Regression
alpha_r = []
error_r = []

for i in range(100,300):
    alpha_r.append(i/10)
    lm_r = Ridge(alpha=i/10)
    error_r.append(np.mean(cross_val_score(lm_r, X_train, y_train, scoring = 'neg_mean_absolute_error')))
    
plt.plot(alpha_r,error_r)
plt.savefig('img/ridge_alpha-vs-error.png')

ridge_parameters = {'alpha':[i/10 for i in range(150, 300)]}
gs_ridge = GridSearchCV(lm_r, ridge_parameters, scoring='neg_mean_absolute_error')
gs_ridge.fit(X_train, y_train)
gs_ridge.best_score_
gs_ridge.best_estimator_

# Elastic Net Regression
elastic_net = ElasticNet()
elastic_net.fit(X_train, y_train)
np.mean(cross_val_score(elastic_net, X_train, y_train, scoring = 'neg_mean_absolute_error'))

# Hyper-parameter tuning for Elastic Net
alpha_en = []
error_en = []

for i in range(1,100):
    alpha_en.append(i/10000)
    elastic_net = ElasticNet(alpha = i/10000)
    error_en.append(np.mean(cross_val_score(elastic_net, X_train, y_train, scoring = 'neg_mean_absolute_error')))
    
plt.plot(alpha_en,error_en)
plt.savefig('img/elasticnet_alpha-vs-error.png')

en_parameters = {'l1_ratio':[i/10 for i in range(1, 10)], 'alpha':[i/10000 for i in range(1, 20)]}
gs_en = GridSearchCV(elastic_net, en_parameters, scoring='neg_mean_absolute_error')
gs_en.fit(X_train, y_train)
gs_en.best_score_
gs_en.best_estimator_

# Compute MSE, MAE, R^2 of the 3 models
print("Best Lasso model: ", gs_lasso.best_estimator_)
print("Best Ridge model: ", gs_ridge.best_estimator_)
print("Elatisc Net best model: ", gs_en.best_estimator_)

# Test ensembles 
tpred_lm = lm.predict(X_test)
tpred_lasso = gs_lasso.best_estimator_.predict(X_test)
tpred_ridge = gs_ridge.best_estimator_.predict(X_test)
tpred_en = gs_en.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("MAE of Lasso: ", mean_absolute_error(y_test, np.exp(tpred_lasso)))
print("MAE of Ridge: ", mean_absolute_error(y_test, np.exp(tpred_ridge)))
print("MAE of Elastic Net: ", mean_absolute_error(y_test, np.exp(tpred_en)))

print("R2 of Lasso: ", r2_score(y_test, np.exp(tpred_lasso)))
print("R2 of Ridge: ", r2_score(y_test, np.exp(tpred_ridge)))
print("R2 of Elastic Net: ", r2_score(y_test, np.exp(tpred_en)))

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

# Through evaluating MSE, MAE and R^2, it is clear that Ridge works best among the four models
chosen_model = Ridge(alpha=24.1)
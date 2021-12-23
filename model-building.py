from feature-engineering import final_df
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

target_column = "SalePrice"

# Split data into training and testing set
X_train, X_val, y_train, y_val = train_test_split(
    df.drop(columns=target_column), final_df[target_column]
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

print("MAE of Lasso: ", mean_absolute_error(y_test, np.exp(tpred_lasso)))
print("MAE of Ridge: ", mean_absolute_error(y_test, np.exp(tpred_ridge)))
print("MAE of Elastic Net: ", mean_absolute_error(y_test, np.exp(tpred_en)))

print("R2 of Lasso: ", r2_score(y_test, np.exp(tpred_lasso)))
print("R2 of Ridge: ", r2_score(y_test, np.exp(tpred_ridge)))
print("R2 of Elastic Net: ", r2_score(y_test, np.exp(tpred_en)))

# XG Boost Regressor
xg_reg = xgb.XGBRegressor()
xg_reg.fit(X_train,y_train)
joblib.dump(xg_reg, "xgreg.pkl")

# Hyperparameter tuning
# Hyperparameter tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }

def objective(space):
    reg = xgb.XGBRegressor(n_estimators =space['n_estimators'], 
                           max_depth = int(space['max_depth']),
                           gamma = space['gamma'], 
                           reg_alpha = int(space['reg_alpha']),
                           min_child_weight=int(space['min_child_weight']),
                           colsample_bytree=int(space['colsample_bytree']))

    eval_set  = [(X_train, y_train), (X_val, y_val)]

    reg.fit(X_train, y_train, eval_set=eval_set, eval_metric = 'rmse',
            early_stopping_rounds=10,verbose=False)
    val_pred = reg.predict(X_val)
    mse = mean_squared_error(y_val, val_pred)
    return{'loss':mse, 'status': STATUS_OK }

trials = Trials()
best_hyperparams = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print(best_hyperparams)

print("The best hyperparameters are : ","\n")
print(best_hyperparams)
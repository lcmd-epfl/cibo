import numpy as np
from datasets import Evaluation_data
from exp_configs_1 import benchmark
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pdb
import deepchem as dc
import bolift
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from botorch_ext import XGBoostSurrogate
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import xgboost as xgb


exp_config = benchmark[0]

DATASET = Evaluation_data(
    exp_config["dataset"],
    exp_config["ntrain"],
    "random",
    init_strategy=exp_config["init_strategy"],
)
bounds_norm = DATASET.bounds_norm

(
    X_init,
    y_init,
    costs_init,
    X_candidate,
    y_candidate,
    costs_candidate,
) = DATASET.get_init_holdout_data(777)

param_grid = {
    "n_estimators": [20, 100, 200],  # Number of boosting rounds (you can adjust this)
    "learning_rate": [2e-2,1e-1, 1],  # Step size shrinkage to prevent overfitting
    "max_depth": [10, 20, 30],  # Maximum tree depth (you can adjust this)
    "min_child_weight": [
        5
    ],  # Minimum sum of instance weight (Hessian) needed in a child
    "subsample": [0.5],  # Fraction of samples used for training
    "colsample_bytree": [1],  # Fraction of features used for training
    "gamma": [
        0
    ],  # Minimum loss reduction required to make a further partition on a leaf node
    "reg_alpha": [0.5, 1, 1.5],  # L1 regularization term on weights
    "reg_lambda": [0.25, 0.5, 1],  # L2 regularization term on weights
}


model = xgb.XGBRegressor(**param_grid)
stratified_kfold = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=stratified_kfold,
    verbose=1,
    n_jobs=-1,
    scoring="neg_root_mean_squared_error",  # Regression task",
)
grid_search.fit(X_init, y_init)
print(grid_search.best_params_)

model = XGBoostSurrogate(grid_search.best_estimator_)
predictions = model.posterior(torch.tensor(X_candidate))
y_pred = model.posterior(torch.tensor(X_candidate)).mean.flatten()
y_std = np.sqrt(model.posterior(torch.tensor(X_candidate)).variance).flatten()

plt.errorbar(y_candidate, y_pred, yerr=y_std, marker=None, fmt=",", alpha=0.6)
plt.plot(y_candidate, y_candidate, color="black", alpha=0.5)
plt.scatter(y_candidate, y_pred, c=y_std, alpha=0.6)
plt.colorbar()
plt.xlabel("EXPERIMENT")
plt.ylabel("PREDICTION")
# compute r2 using sklearn and RMSE
r2 = r2_score(y_pred, y_candidate)
mae = mean_absolute_error(y_candidate, y_pred)
print("r2 = ", r2)
print("N = ", len(X_init), "MAE = ", mae)

plt.savefig("correlation.png")
# pdb.set_trace()

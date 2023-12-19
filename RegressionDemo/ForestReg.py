import numpy as np
from datasets import Evaluation_data
from exp_configs_1 import benchmark
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import torch
from sklearn.ensemble import RandomForestRegressor
from botorch_ext import ForestSurrogate
import pdb

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


model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_init, y_init)
mod = False

model = ForestSurrogate(model)
predictions = model.posterior(torch.tensor(X_candidate))
y_pred = model.posterior(torch.tensor(X_candidate)).mean.flatten()
y_std = np.sqrt(model.posterior(torch.tensor(X_candidate)).variance).flatten()

plt.errorbar(y_candidate, y_pred, yerr=y_std, marker=None, fmt=",", alpha=0.6)
plt.plot(y_candidate, y_candidate, color="black", alpha=0.5)
plt.scatter(y_candidate, y_pred, c=y_std, alpha=0.6)

plt.xlabel("EXPERIMENT")
plt.ylabel("PREDICTION")
# compute r2 using sklearn and RMSE
r2 = r2_score(y_pred, y_candidate)
mae = mean_absolute_error(y_candidate, y_pred)
print("r2 = ", r2)
print("N = ", len(X_init), "MAE = ", mae)

plt.savefig("correlation.png")
# pdb.set_trace()

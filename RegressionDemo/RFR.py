import numpy as np
from datasets import Evaluation_data
from exp_configs_1 import benchmark
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import torch
from sklearn.ensemble import RandomForestRegressor
from botorch_ext import ForestSurrogate
import random

np.random.seed(777)
random.seed(777)

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


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_init, y_init)
mod = False

model = ForestSurrogate(model)
predictions = model.posterior(torch.tensor(X_candidate))
y_pred = model.posterior(torch.tensor(X_candidate)).mean.flatten()
y_std = np.sqrt(model.posterior(torch.tensor(X_candidate)).variance).flatten()


r2 = r2_score(y_pred, y_candidate)
mae = mean_absolute_error(y_candidate, y_pred)

print("r2 = ", r2)
print("N = ", len(X_init), "MAE = ", mae)


fig, ax = plt.subplots(1, 1, figsize=(5, 5))


if exp_config["dataset"] == "BMS":
    dataset = "C-H Acrylation"
elif exp_config["dataset"] == "buchwald":
    dataset = "Buchwald"
elif exp_config["dataset"] == "baumgartner":
    dataset = "Baumgartner"
else:
    raise ValueError("Unknown dataset")

ax.set_title("{} RFR".format(dataset))
ax.errorbar(y_candidate, y_pred, yerr=y_std, marker=None, fmt=",", alpha=0.1)
ax.plot(y_candidate, y_candidate, color="black", alpha=0.2)
ax.scatter(y_candidate, y_pred, alpha=0.6)
# Setting the axis limits
ax.set_xlim(0, 102)
ax.set_ylim(0, 102)
ax.text(
    5, 95, f"R² score: {r2:.2f}", fontsize=12
)  # Adjust position and fontsize as needed
ax.text(5, 90, f"MAE: {mae:.2f}", fontsize=12)  # Adjust position and fontsize as needed
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.tick_params(axis="both", which="major", labelsize=12)
ax.set_xlabel("Experiment", fontsize=12, fontweight="bold")
ax.set_ylabel("Prediction", fontsize=12, fontweight="bold")
plt.savefig(f"correlation_{dataset}_RFR.pdf")

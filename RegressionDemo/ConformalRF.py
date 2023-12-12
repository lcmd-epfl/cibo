import numpy as np
import torch
from BO import update_model
from utils import check_entries, convert2pytorch
from datasets import Evaluation_data
from exp_configs_1 import benchmark
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from mapie.conformity_scores import GammaConformityScore
from mapie.metrics import regression_coverage_score
from mapie.regression import MapieRegressor
import pdb


def get_yerr(y_pred, y_pis):
    return np.concatenate(
        [
            np.expand_dims(y_pred, 0) - y_pis[:, 0, 0].T,
            y_pis[:, 1, 0].T - np.expand_dims(y_pred, 0),
        ],
        axis=0,
    )


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


pdb.set_trace
alpha = 0.05
rf_kwargs = {"n_estimators": 22, "random_state": 2}
RF = RandomForestRegressor(**rf_kwargs)


mapie = MapieRegressor(RF, conformity_score=GammaConformityScore(), method="plus", cv=5, agg_function="median", n_jobs=-1, random_state=2)

mapie.fit(X_init, y_init + 0.0001)
y_pred_gammaconfscore, y_pis_gammaconfscore = mapie.predict(
    X_candidate, alpha=[alpha], ensemble=True
)


yerr_gammaconfscore = get_yerr(y_pred_gammaconfscore, y_pis_gammaconfscore)
y_std = yerr_gammaconfscore[1] - yerr_gammaconfscore[0]
#pdb.set_trace()
y_pred = y_pred_gammaconfscore
y_candidate = y_candidate.numpy().flatten()

plt.errorbar(y_candidate, y_pred, yerr=y_std, marker=None, fmt="o", alpha=0.2)
plt.plot(y_candidate, y_candidate, color="black", alpha=0.5)
#plt.scatter(y_candidate, y_pred, c=y_std, alpha=0.6)
# add colorbar with
#plt.colorbar()
plt.xlabel("EXPERIMENT")
plt.ylabel("PREDICTION")


# compute r2 using sklearn and RMSE
r2 = r2_score(y_pred, y_candidate)
mae = mean_absolute_error(y_candidate, y_pred)
print("r2 = ", r2)
print("N = ", len(X_init), "MAE = ", mae)

plt.savefig("correlation.png")

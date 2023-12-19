import numpy as np
from datasets import Evaluation_data
from exp_configs_1 import benchmark
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from mapie.conformity_scores import GammaConformityScore, AbsoluteConformityScore
from mapie.regression import MapieRegressor
import pdb

from mapie.metrics import regression_coverage_score


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


mapie = MapieRegressor(
    RF,
    conformity_score=GammaConformityScore(),
    method="plus",
    cv=5,
    agg_function="median",
    n_jobs=-1,
    random_state=2,
)

mapie.fit(X_init, y_init + 1)
y_pred_gammaconfscore, y_pis_gammaconfscore = mapie.predict(
    X_candidate, alpha=[alpha], ensemble=True
)


coverage_gammaconfscore = regression_coverage_score(
    y_candidate, y_pis_gammaconfscore[:, 0, 0], y_pis_gammaconfscore[:, 1, 0]
)

y_err_gammaconfscore = get_yerr(y_pred_gammaconfscore, y_pis_gammaconfscore)
# pdb.set_trace()
y_std = y_err_gammaconfscore  # [1] - y_err_gammaconfscore[0]
# pdb.set_trace()
y_pred = y_pred_gammaconfscore
y_candidate = y_candidate.numpy().flatten()

plt.errorbar(y_candidate, y_pred, yerr=y_std, marker=None, fmt="o", alpha=0.2)
plt.plot(y_candidate, y_candidate, color="black", alpha=0.5)

plt.xlabel("EXPERIMENT")
plt.ylabel("PREDICTION")


# compute r2 using sklearn and RMSE
r2 = r2_score(y_pred, y_candidate)
mae = mean_absolute_error(y_candidate, y_pred)
print("r2 = ", r2)
print("N = ", len(X_init), "MAE = ", mae)

plt.savefig("correlation.png")

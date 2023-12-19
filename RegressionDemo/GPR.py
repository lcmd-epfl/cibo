import numpy as np
from BO import update_model
from datasets import Evaluation_data
from exp_configs_1 import benchmark
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


import matplotlib.pyplot as plt


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


fit_y = False

model, scaler_y = update_model(
    X_init, y_init, bounds_norm, kernel_type="Tanimoto", fit_y=fit_y, FIT_METHOD=True
)


if fit_y:
    y_pred = scaler_y.inverse_transform(
        model.posterior(X_candidate).mean.detach()
    ).flatten()
    y_std = np.sqrt(
        scaler_y.inverse_transform(
            model.posterior(X_candidate).variance.detach()
        ).flatten()
    )

else:
    y_pred = model.posterior(X_candidate).mean.detach().flatten().numpy()
    y_std = np.sqrt(model.posterior(X_candidate).variance.detach().flatten().numpy())

y_pred = y_pred.flatten()
y_candidate = y_candidate.numpy().flatten()

plt.errorbar(y_candidate, y_pred, yerr=y_std,marker=None,  fmt=",", alpha=0.6)
plt.plot(y_candidate, y_candidate, color="black", alpha=0.5)
plt.scatter(y_candidate, y_pred,c=y_std, alpha=0.6)
#add colorbar with 
plt.colorbar()
plt.xlabel("EXPERIMENT")
plt.ylabel("PREDICTION")


# compute r2 using sklearn and RMSE
r2 = r2_score(y_pred, y_candidate)
mae = mean_absolute_error(y_candidate, y_pred)
print("r2 = ", r2)
print("N = ", len(X_init), "MAE = ", mae)

plt.savefig("correlation.png")

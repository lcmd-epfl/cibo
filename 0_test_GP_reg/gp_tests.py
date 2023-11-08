from BO import *
from utils import *

RESULTS = []

for exp_config in benchmark:
    DATASET = Evaluation_data(
        exp_config["dataset"],
        exp_config["ntrain"],
        "random",
        init_strategy=exp_config["init_strategy"],
    )
    bounds_norm = DATASET.bounds_norm
pred = model.posterior(X_candidate_FULL).mean.detach().flatten().numpy()

pred = (
                        scaler_y.inverse_transform(
                            BO_data["model"]
                            .posterior(BO_data["X_candidate_BO"])
                            .mean.detach()
                        )
                        .flatten()
                        .numpy()
                    )
                    # var = np.sqrt(model.posterior(X_candidate_FULL).variance.detach().flatten().numpy())
                    # scaler_y.inverse_transform(model.posterior(X_candidate_FULL).mean.detach()).flatten().numpy()
                    y_test = BO_data[
                        "y_candidate_BO"
                    ].flatten()  # y_candidate_FULL.numpy().flatten()
                    plt.clf()
                    plt.close()
                    plt.scatter(pred, y_test, marker="o")
                    plt.savefig("shit.png")

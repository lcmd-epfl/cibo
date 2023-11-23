from BO import *
from utils import *
from exp_configs_1 import benchmark
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import deepchem as dc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

TEST_OPTION = 2

if TEST_OPTION == 1:
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

else:

    featurizer = dc.feat.CircularFingerprint(size=512)

    tasks, datasets, transformers = dc.molnet.load_sampl(
        featurizer=featurizer, splitter="random", transformers=[]
    )
    train_dataset, valid_dataset, test_dataset = datasets

    # Extract training data from DeepChem dataset, and convert to NumPy arrays
    X_train = train_dataset.X

    bounds_norm = torch.tensor([[0] * 512, [1] * 512])
    bounds_norm = bounds_norm.to(dtype=torch.float32)

    if not check_entries(X_train):
        X_train = MinMaxScaler().fit_transform(X_train)

    y_train = train_dataset.y[:, 0].reshape(-1, 1)

    # pdb.set_trace()
    # Split the data into training and test sets
    X_init, X_candidate, y_init, y_candidate = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    X_init, y_init = convert2pytorch(X_init, y_init)
    X_candidate, y_candidate = convert2pytorch(X_candidate, y_candidate)


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

plt.errorbar(y_candidate, y_pred, yerr=y_std, fmt="o", alpha=0.6)
plt.plot(y_candidate, y_candidate, color="black", alpha=0.5)
plt.xlabel("EXPERIMENT")
plt.ylabel("PREDICTION")


# compute r2 using sklearn and RMSE
r2 = r2_score(y_candidate, y_pred)
rmse = mean_absolute_error(y_candidate, y_pred)
print("r2 = ", r2)
print("N = ", len(X_init), "MAE = ", rmse)

plt.savefig("correlation.png")

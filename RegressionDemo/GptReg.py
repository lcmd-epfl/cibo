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
from rdkit import Chem
import pubchempy as pcp
from tqdm import tqdm


def smiles_to_iupac(smiles_list):
    iupac_names = []

    for smiles in tqdm(smiles_list):
        try:
            compounds = pcp.get_compounds(smiles, "smiles")
            # Assuming the first returned compound is the desired one
            if compounds:
                iupac_name = compounds[0].iupac_name
                iupac_names.append(iupac_name)
            else:
                # iupac_names.append("No match found")
                iupac_names.append(smiles)
        except Exception as e:
            iupac_names.append(smiles)

    return iupac_names


# pdb.set_trace()


def get_asktell(model: str, kwargs: dict = {}, pool: bolift.Pool = None, knn: int = 1):
    if model == "instruct":
        kwargs["model"] = "gpt-3.5-turbo-instruct"
        return bolift.AskTellFewShotTopk(**kwargs)
    elif model == "gpt-turbo":
        kwargs["model"] = "gpt-3.5-turbo"
        return bolift.AskTellFewShotTopk(**kwargs)
    elif model == "gpt-4":
        kwargs["model"] = "gpt-4"
        return bolift.AskTellFewShotTopk(**kwargs)
    elif model == "davinci":
        kwargs["model"] = "text-davinci-003"
        return bolift.AskTellFewShotTopk(**kwargs)
    elif model == "curie":
        kwargs["model"] = "text-curie-001"
        return bolift.AskTellFewShotTopk(**kwargs)
    elif model == "gpr":
        kwargs["selector_k"] = 0
        kwargs["pool"] = pool if pool else None
        kwargs["n_components"] = 32
        return bolift.AskTellGPR(**kwargs)
    # Uncomment and adjust the following lines as needed
    # elif model == "knn":
    #     del kwargs['selector_k']
    #     kwargs['knn'] = knn
    #     return bolift.AskTellNearestNeighbor(**kwargs)
    # elif model == "krr":
    #     kwargs['alpha'] = 0.5
    #     return bolift.AskTellRidgeKernelRegression(**kwargs)
    else:
        raise ValueError("Unknown model")


def train(model, SMILES, values):
    for smi, v in zip(SMILES, values):
        model.tell(smi, v)
    return model


featurizer = dc.feat.CircularFingerprint(size=512)

tasks, datasets, transformers = dc.molnet.load_sampl(
    featurizer=featurizer, splitter="random", transformers=[]
)
train_dataset, valid_dataset, test_dataset = datasets

# Extract training data from DeepChem dataset, and convert to NumPy arrays
X_train = train_dataset.ids

X_train = smiles_to_iupac(X_train)

y_train = train_dataset.y[:, 0]

# pdb.set_trace()
# Split the data into training and test sets
X_init, X_candidate, y_init, y_candidate = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)


model = bolift.AskTellFewShotTopk(
    x_formatter=lambda x: f"iupac name {x}",
    y_name="free energy (kcal/mol)",
    y_formatter=lambda y: f"{y:.2f}",
    model="gpt-3.5-turbo-instruct",
    selector_k=5,
    temperature=0.05,
)
# get_asktell("gpt-4")
pdb.set_trace()
model = train(model, X_init[:250], y_init[:250])

predictions = [model.predict(x) for x in X_candidate]
y_pred = np.array([p.mean() for p in predictions])
y_std = np.array([p.std() for p in predictions])
# print(yhat.mean(), yhat.std())


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
pdb.set_trace()

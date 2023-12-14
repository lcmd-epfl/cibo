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
from botorch_ext import ForestSurrogate
from tqdm import tqdm

featurizer = dc.feat.CircularFingerprint(size=512)

tasks, datasets, transformers = dc.molnet.load_sampl(
    featurizer=featurizer, splitter="random", transformers=[]
)
train_dataset, valid_dataset, test_dataset = datasets

# Extract training data from DeepChem dataset, and convert to NumPy arrays
X_train = train_dataset.X
y_train = train_dataset.y[:, 0]

# pdb.set_trace()
# Split the data into training and test sets
X_init, X_candidate, y_init, y_candidate = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)


model = RandomForestRegressor(n_estimators=30, random_state=42)
model.fit(X_init, y_init)
mod = False

model = ForestSurrogate(model)
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
#pdb.set_trace()

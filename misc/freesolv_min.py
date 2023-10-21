import math
import torch

from botorch.test_functions import SixHumpCamel
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.utils.transforms import standardize, normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
import os
import deepchem as dc
import pdb
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from botorch.utils.transforms import standardize, normalize
from botorch.optim import optimize_acqf, optimize_acqf_discrete
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
#https://botorch.org/tutorials/GIBBON_for_efficient_batch_entropy_search
torch.manual_seed(123456)
import copy as cp

featurizer = dc.feat.CircularFingerprint(size=1024)


# Load FreeSolv dataset
tasks, datasets, transformers = dc.molnet.load_sampl(featurizer=featurizer, splitter='random', transformers = [])
train_dataset, valid_dataset, test_dataset = datasets

# Extract training data from DeepChem dataset
X_train = train_dataset.X
y_train = train_dataset.y[:, 0]

X_valid = valid_dataset.X
y_valid = valid_dataset.y[:, 0]

X_test = test_dataset.X
y_test = test_dataset.y[:, 0]

#connect all numpy arrays into one
X = np.concatenate((X_train, X_valid, X_test))
y = np.concatenate((y_train, y_valid, y_test)) 


index_worst = np.random.choice(np.argwhere(y<2).flatten(), size=500, replace=False)
index_others = np.setdiff1d(np.arange(len(y)), index_worst)
#randomly shuffle the data
index_others = np.random.permutation(index_others)

X_init, y_init = X[index_worst], y[index_worst]
X_candidate, y_candidate = X[index_others], y[index_others]


#convert X_init and X_candidate to torch float 32 tensors and reshape the y tensors
#need to normalize everything together

X_init = torch.from_numpy(X_init).float()
X_candidate = torch.from_numpy(X_candidate).float()
y_init = torch.from_numpy(y_init).float().reshape(-1,1)
y_candidate = torch.from_numpy(y_candidate).float().reshape(-1,1)



bounds_norm = torch.tensor([[0]*1024, [1]*1024])
bounds_norm = bounds_norm.to(dtype=torch.float32)
train_X = normalize(X_init, bounds=bounds_norm)
train_Y  =y_init
train_Y = standardize(y_init)
y_candidate = standardize(y_candidate)
X, y = cp.deepcopy(train_X), cp.deepcopy(y_init)

X_candidate = normalize(X_candidate, bounds=bounds_norm)
y_best = torch.max(y)
model = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll) #, max_attempts=10)


NUM_RESTARTS = 100
RAW_SAMPLES = 100
#pdb.set_trace()
#model.posterior(X_candidate).mean

for i in range(20):
    qGIBBON = qLowerBoundMaxValueEntropy(model, X_candidate)
    candidates, acq_value = optimize_acqf_discrete(acq_function=qGIBBON,bounds=bounds_norm,q=20,choices=X_candidate, num_restarts=NUM_RESTARTS,raw_samples=RAW_SAMPLES,sequential=True)

    #find the indices of the candidates in X_candidate
    indices = []
    for candidate in candidates:
        indices.append(np.argwhere((X_candidate==candidate).all(1)).flatten()[0])

    X, y = np.concatenate((X,candidates)), np.concatenate((y, y_candidate[indices, :]))
    y_best  = max(y)[0]
    model = SingleTaskGP(torch.from_numpy(X).float(), torch.from_numpy(y).float().reshape(-1,1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, max_attempts=10)
    #remove the candidates from X_candidate
    X_candidate = np.delete(X_candidate, indices, axis=0)
    #remove the corresponding y values
    y_candidate = np.delete(y_candidate, indices, axis=0)
    #pdb.set_trace()
    #pdb.set_trace()

    #print(candidates, acq_value)
    print(i, y_best)


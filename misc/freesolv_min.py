import math
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.utils.transforms import standardize, normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
import os
import deepchem as dc
import pdb
import numpy as np
from botorch.utils.transforms import standardize, normalize
from botorch.optim import optimize_acqf, optimize_acqf_discrete
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy, qMaxValueEntropy
#https://botorch.org/tutorials/GIBBON_for_efficient_batch_entropy_search
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from BO import CustomGPModel

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
#multiply the columns number 3 of X by 100
#X[:, 3] = X[:, 3]*10

#pdb.set_trace()
index_worst =np.random.choice(np.argwhere(y<-2).flatten(), size=50, replace=False)
#np.random.choice(np.arange(len(y)), size=500, replace=False)
#
index_others = np.setdiff1d(np.arange(len(y)), index_worst)
#randomly shuffle the data
index_others = np.random.permutation(index_others)

X_init, y_init = X[index_worst], y[index_worst]
X_candidate, y_candidate = X[index_others], y[index_others]





X_init = torch.from_numpy(X_init).float()
X_candidate = torch.from_numpy(X_candidate).float()
y_init = torch.from_numpy(y_init).float().reshape(-1,1)
y_candidate = torch.from_numpy(y_candidate).float().reshape(-1,1)


bounds_norm = torch.tensor([[0]*1024, [1]*1024])
bounds_norm = bounds_norm.to(dtype=torch.float32)

#y_candidate = standardize(y_candidate)
X, y = cp.deepcopy(X_init), cp.deepcopy(y_init)
# multiply the columns number 3 of X by 100
X[:, 0] = X[:, 0]*70
#adapt the bounds to the new X
bounds_norm[1][0] = 70
#pdb.set_trace()
 # normalize(X_candidate, bounds=bounds_norm).to(dtype=torch.float32)
y_best = torch.max(y)


GP_class = CustomGPModel(kernel_type="Matern", scale_type_X="botorch", bounds_norm=bounds_norm)
#(Pdb) GP_class.scaler_X.data_min_
#array([0., 0., 0., ..., 0., 0., 0.])
#(Pdb) GP_class.scaler_X.data_max_
#array([1., 1., 1., ..., 1., 1., 1.])
model = GP_class.fit(X, y)
#pdb.set_trace()
#in case sklearn scaler is used
# X_candidate =    torch.from_numpy(GP_class.scaler_X.transform(X_candidate.detach().numpy()))
#pdb.set_trace()

X_candidate = normalize(X_candidate, bounds=bounds_norm).to(dtype=torch.float32)
pred = GP_class.scaler_y.inverse_transform(model.posterior(X_candidate).mean.detach().numpy())

#make a scatter plot of the predictions
import matplotlib.pyplot as plt

X_candidate_FULL, y_candidate_FULL = cp.deepcopy(X_candidate), cp.deepcopy(y_candidate)

#bounds_norm = torch.tensor([GP_class.scaler_X.data_min_.tolist(), GP_class.scaler_X.data_max_.tolist()])
#bounds_norm = bounds_norm.to(dtype=torch.float32)


NUM_RESTARTS = 10
RAW_SAMPLES = 512

#model.posterior(X_candidate).mean
y_better_BO = []
y_better_RANDOM = []

for i in range(10):
    qGIBBON = qLowerBoundMaxValueEntropy(model, X_candidate)
    candidates, acq_value = optimize_acqf_discrete(acq_function=qGIBBON,bounds=bounds_norm,q=20,choices=X_candidate, num_restarts=NUM_RESTARTS,raw_samples=RAW_SAMPLES,sequential=True)

    #find the indices of the candidates in X_candidate
    indices = []
    for candidate in candidates:
        indices.append(np.argwhere((X_candidate==candidate).all(1)).flatten()[0])

    X, y = np.concatenate((X,candidates)), np.concatenate((y, y_candidate[indices, :]))
    y_best  = max(y)[0]
    GP_class = CustomGPModel(kernel_type="Matern", scale_type_X="botorch", bounds_norm=bounds_norm)
    model = GP_class.fit(X, y)



    X_candidate = np.delete(X_candidate, indices, axis=0)
    y_candidate = np.delete(y_candidate, indices, axis=0)

    pred = GP_class.scaler_y.inverse_transform(model.posterior(X_candidate_FULL).mean.detach().numpy())

    
    plt.scatter(pred, y_candidate_FULL)
    plt.show()


    #print(candidates, acq_value)
    print(i, y_best)


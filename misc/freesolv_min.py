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
from botorch.acquisition import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    qMaxValueEntropy,
    UpperConfidenceBound
)
#https://botorch.org/tutorials/GIBBON_for_efficient_batch_entropy_search
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from BO import CustomGPModel

torch.manual_seed(34634)
import copy as cp

featurizer = dc.feat.RDKitDescriptors()
#dc.feat.CircularFingerprint(size=1024)


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



index_worst =np.random.choice(np.argwhere(y<-2).flatten(), size=300, replace=False)
index_others = np.setdiff1d(np.arange(len(y)), index_worst)
#randomly shuffle the data
index_others = np.random.permutation(index_others)

X_init, y_init = X[index_worst], y[index_worst]
X_candidate, y_candidate = X[index_others], y[index_others]



X_init = torch.from_numpy(X_init).float()
X_candidate = torch.from_numpy(X_candidate).float()
#pdb.set_trace()
y_init = torch.from_numpy(y_init).float().reshape(-1,1)
y_candidate = torch.from_numpy(y_candidate).float().reshape(-1,1)


bounds_norm = torch.tensor([torch.min(X_candidate, dim=0).values.tolist(),torch.max(X_candidate, dim=0).values.tolist()])
# add 0.5 to the max value
bounds_norm[1] = bounds_norm[1] + 1.0
#multiply the max value by 2
bounds_norm[1] = bounds_norm[1]*2


bounds_norm = bounds_norm.to(dtype=torch.float32)


X, y = cp.deepcopy(X_init), cp.deepcopy(y_init)


y_best = torch.max(y)


GP_class = CustomGPModel(kernel_type="Matern", scale_type_X="botorch", bounds_norm=bounds_norm)

model = GP_class.fit(X, y)

X_candidate = normalize(X_candidate, bounds=bounds_norm).to(dtype=torch.float32)


pred = model.posterior(X_candidate).mean.detach().numpy()
pred_error = model.posterior(X_candidate).variance.sqrt().detach().numpy()
#pred = GP_class.scaler_y.inverse_transform(model.posterior(X_candidate).mean.detach().numpy())
#pred_error = np.abs( GP_class.scaler_y.inverse_transform(model.posterior(X_candidate).variance.sqrt().detach().numpy()))

#make a scatter plot of the predictions
import matplotlib.pyplot as plt
#plot the predictions and error bars
pdb.set_trace()
plt.scatter(pred, y_candidate)
plt.errorbar(pred.flatten(), y_candidate.flatten(), yerr=pred_error.flatten(), fmt='o')

plt.show()



X_candidate_FULL, y_candidate_FULL = cp.deepcopy(X_candidate), cp.deepcopy(y_candidate)


y_candidate_RANDOM = cp.deepcopy(y_candidate).detach().numpy()



NUM_RESTARTS = 20
RAW_SAMPLES = 512

y_better_BO = []
y_better_RANDOM = []


y_better_BO.append(y_best)
y_better_RANDOM.append(y_best)

y_best_BO, y_best_RANDOM = y_best, y_best


NITER = 10
for i in range(NITER):
    qGIBBON = qLowerBoundMaxValueEntropy(model,X_candidate)
    #UpperConfidenceBound(model, beta=2)
    #qLowerBoundMaxValueEntropy(model,X_candidate)
    #qLowerBoundMaxValueEntropy(model,X_candidate ) 
    #ExpectedImprovement(model, best_f=y_best_BO)
    #qLowerBoundMaxValueEntropy(model,X_candidate ) 
    #ExpectedImprovement(model, best_f=y_best_BO)
    #qLowerBoundMaxValueEntropy(model,X_candidate ) 
    
    #qGIBBON = ExpectedImprovement(model, best_f=y_best_BO)
    candidates, acq_value = optimize_acqf_discrete(acq_function=qGIBBON,bounds=bounds_norm,q=1,choices=X_candidate, num_restarts=NUM_RESTARTS,raw_samples=RAW_SAMPLES,sequential=True)

    #find the indices of the candidates in X_candidate
    indices = []
    for candidate in candidates:
        indices.append(np.argwhere((X_candidate==candidate).all(1)).flatten()[0])

    #pdb.set_trace()
    X, y = np.concatenate((X,candidates)), np.concatenate((y, y_candidate[indices, :]))
    if max(y)[0] > y_best_BO:
        y_best_BO = max(y)[0]

    y_better_BO.append(y_best_BO)

    GP_class = CustomGPModel(kernel_type="Matern", scale_type_X="botorch", bounds_norm=bounds_norm)
    model = GP_class.fit(X, y)

    X_candidate = np.delete(X_candidate, indices, axis=0)
    y_candidate = np.delete(y_candidate, indices, axis=0)

    pred = model.posterior(X_candidate_FULL).mean.detach().numpy()
    #GP_class.scaler_y.inverse_transform(

    
    #plt.scatter(pred, y_candidate_FULL)
    #plt.show()
    #plt.clf()
    ##pick a random value from y_candidate_RANDOM
    index = np.random.choice(np.arange(len(y_candidate_RANDOM)))
    #pdb.set_trace()
    if y_candidate_RANDOM[index][0] > y_best_RANDOM:
        y_best_RANDOM = y_candidate_RANDOM[index][0]

    y_better_RANDOM.append(y_best_RANDOM)

    y_candidate_RANDOM = np.delete(y_candidate_RANDOM, index, axis=0)


    #print(candidates, acq_value)
    print(i, y_best_BO, y_best_RANDOM)

#pdb.set_trace()
#compare the two plots
fig, ax = plt.subplots()
#pdb.set_trace()
ax.plot(np.arange(NITER+1) , y_better_BO, label="BO")
ax.plot(np.arange(NITER+1) , y_better_RANDOM, label="RANDOM")
ax.legend()
plt.show()
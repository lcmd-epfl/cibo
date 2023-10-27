import torch
from botorch.utils.transforms import normalize

import pdb
import numpy as np
from botorch.optim import optimize_acqf_discrete
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
#https://botorch.org/tutorials/GIBBON_for_efficient_batch_entropy_search
import matplotlib.pyplot as plt
import random
import copy as cp
from BO import *
from utils import *
from sklearn.model_selection import train_test_split


FREESOLV = Evaluation_data("freesolv", 30, "random",init_strategy="random")
bounds_norm = FREESOLV.bounds_norm


N_RUNS = 10
NITER = 10
NUM_RESTARTS = 40
RAW_SAMPLES = 512
BATCH_SIZE = 1 #2
SEQUENTIAL = False
y_better_BO_ALL, y_better_RANDOM_ALL = [], []
running_costs_BO_ALL, running_costs_RANDOM_ALL = [], []

MAX_BATCH_COST = 0

COST_AWARE_BO = False
COST_AWARE_RANDOM = False

for run in range(N_RUNS):
    random.seed(111+run)
    torch.manual_seed(111+run)

    #pdb.set_trace()
    X_init, y_init, costs_init, X_candidate, y_candidate, costs_candidate = FREESOLV.get_init_holdout_data()
    X, y = cp.deepcopy(X_init), cp.deepcopy(y_init)
    y_best = float(torch.max(y))
    model, scaler_y = update_model(X, y, bounds_norm)


    X_candidate = normalize(X_candidate, bounds=bounds_norm).to(dtype=torch.float32)
    X_candidate_FULL, y_candidate_FULL = cp.deepcopy(X_candidate), cp.deepcopy(y_candidate)
    pred = scaler_y.inverse_transform(model.posterior(X_candidate_FULL).mean.detach().numpy())
    #model.posterior(X_candidate_FULL).mean.detach().numpy()
    #
    pred_error = model.posterior(X_candidate).variance.sqrt().detach().numpy()
    


    plt.scatter(pred, y_candidate_FULL, label='Full data')
    plt.savefig("preds.png")
    #exit()



    costs_FULL          = cp.deepcopy(costs_candidate)
    X_candidate_BO      = cp.deepcopy(X_candidate)
    y_candidate_BO      = cp.deepcopy(y_candidate)
    y_candidate_RANDOM  = cp.deepcopy(y_candidate).detach().numpy()


    running_costs_BO = [0]
    running_costs_RANDOM = [0]

    costs_BO        = cp.deepcopy(costs_candidate)
    costs_RANDOM    = cp.deepcopy(costs_candidate)

    y_better_BO = []
    y_better_RANDOM = []

    y_better_BO.append(y_best)
    y_better_RANDOM.append(y_best)
    y_best_BO, y_best_RANDOM = y_best, y_best


    for i in range(NITER):
        if COST_AWARE_BO == False:
            qGIBBON = qLowerBoundMaxValueEntropy(model,X_candidate_BO)
            candidates, acq_value = optimize_acqf_discrete(acq_function=qGIBBON,bounds=bounds_norm,q=BATCH_SIZE,choices=X_candidate_BO, num_restarts=NUM_RESTARTS,raw_samples=RAW_SAMPLES,sequential=SEQUENTIAL)
            indices = find_indices(X_candidate_BO, candidates)
            X, y = np.concatenate((X,candidates)), np.concatenate((y, y_candidate_BO[indices, :]))
            
            if max(y)[0] > y_best_BO:
                y_best_BO = max(y)[0]

            y_better_BO.append(y_best_BO)
            running_costs_BO.append((running_costs_BO[-1] + sum(costs_BO[indices]))[0])
            model, _ = update_model(X, y, bounds_norm)
            X_candidate_BO = np.delete(X_candidate_BO, indices, axis=0)
            y_candidate_BO = np.delete(y_candidate_BO, indices, axis=0)
            costs_BO = np.delete(costs_BO, indices, axis=0)            

        else:    
            qGIBBON = qLowerBoundMaxValueEntropy(model,X_candidate_BO)
            candidates, acq_value = optimize_acqf_discrete(acq_function=qGIBBON,bounds=bounds_norm,q=2*BATCH_SIZE,choices=X_candidate_BO, num_restarts=NUM_RESTARTS,raw_samples=RAW_SAMPLES,sequential=SEQUENTIAL)
            indices = find_indices(X_candidate_BO, candidates)
            suggested_costs = costs_BO[indices].flatten()
            cheap_indices   = select_batch(suggested_costs, MAX_BATCH_COST, BATCH_SIZE)
            cheap_indices = indices[cheap_indices]
            SUCCESS = True
            ITERATION = 1

            while (cheap_indices is None) or (len(cheap_indices) < BATCH_SIZE):
                INCREMENTED_MAX_BATCH_COST = MAX_BATCH_COST
                SUCCESS = False

                INCREMENTED_BATCH_SIZE = 2*BATCH_SIZE + ITERATION
                print("Incrementing batch size to: ", INCREMENTED_BATCH_SIZE)
                if INCREMENTED_BATCH_SIZE > len(X_candidate_BO):
                    print("WTF")
                    break
                    INCREMENTED_MAX_BATCH_COST  += 1

                candidates, acq_value = optimize_acqf_discrete(acq_function=qGIBBON,bounds=bounds_norm,q=INCREMENTED_BATCH_SIZE,choices=X_candidate_BO, num_restarts=NUM_RESTARTS,raw_samples=RAW_SAMPLES,sequential=SEQUENTIAL)

                indices = find_indices(X_candidate_BO, candidates)
                suggested_costs = costs_BO[indices].flatten()
                cheap_indices   = select_batch(suggested_costs, INCREMENTED_MAX_BATCH_COST, BATCH_SIZE)
                cheap_indices   = indices[cheap_indices]

                if cheap_indices is not None and len(cheap_indices) == BATCH_SIZE:
                    X, y = np.concatenate((X,X_candidate_BO[cheap_indices])), np.concatenate((y, y_candidate_BO[cheap_indices, :]))
                    if max(y)[0] > y_best_BO:
                        y_best_BO = max(y)[0]

                    y_better_BO.append(y_best_BO)
                    BATCH_COST = sum(costs_BO[cheap_indices])[0]
                    print("Batch cost: ", BATCH_COST)
                    running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
                    model, scaler_y = update_model(X, y, bounds_norm)
                
                    pred = scaler_y.inverse_transform(model.posterior(X_candidate_FULL).mean.detach().numpy())
                    pred_error = model.posterior(X_candidate).variance.sqrt().detach().numpy()
                    plt.scatter(pred, y_candidate_FULL, label='Full data')


                    X_candidate_BO = np.delete(X_candidate_BO, cheap_indices, axis=0)
                    y_candidate_BO = np.delete(y_candidate_BO, cheap_indices, axis=0)
                    costs_BO = np.delete(costs_BO, cheap_indices, axis=0)
                    break
                
                ITERATION +=1

            if SUCCESS:
                X, y = np.concatenate((X,X_candidate_BO[cheap_indices])), np.concatenate((y, y_candidate_BO[cheap_indices, :]))
                if max(y)[0] > y_best_BO:
                    y_best_BO = max(y)[0]

                y_better_BO.append(y_best_BO)
                BATCH_COST = sum(costs_BO[cheap_indices])[0]
                print("Batch cost: ", BATCH_COST)
                running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
                model,scaler_y = update_model(X, y, bounds_norm)
                X_candidate_BO = np.delete(X_candidate_BO, cheap_indices, axis=0)
                y_candidate_BO = np.delete(y_candidate_BO, cheap_indices, axis=0)
                costs_BO = np.delete(costs_BO, cheap_indices, axis=0)

        if COST_AWARE_RANDOM == False:
            indices_random = np.random.choice(np.arange(len(y_candidate_RANDOM)), size=BATCH_SIZE, replace=False)
        
            if max(y_candidate_RANDOM[indices_random])[0] > y_best_RANDOM:
                y_best_RANDOM = max(y_candidate_RANDOM[indices_random])[0] 

            y_better_RANDOM.append(y_best_RANDOM)
            BATCH_COST = sum(costs_RANDOM[indices_random])[0]
            running_costs_RANDOM.append(running_costs_RANDOM[-1] + BATCH_COST)
            y_candidate_RANDOM = np.delete(y_candidate_RANDOM, indices_random, axis=0)
            costs_RANDOM = np.delete(costs_RANDOM, indices_random, axis=0)
            print(i, y_best_BO, y_best_RANDOM)
        else:
            all_cheapest_indices = np.argwhere(costs_RANDOM.flatten() == 0).flatten()
            indices_random = np.random.choice(all_cheapest_indices, size=BATCH_SIZE, replace=False)
            if max(y_candidate_RANDOM[indices_random])[0] > y_best_RANDOM:
                y_best_RANDOM = max(y_candidate_RANDOM[indices_random])[0]

            y_better_RANDOM.append(y_best_RANDOM)
            BATCH_COST = sum(costs_RANDOM[indices_random])[0]
            running_costs_RANDOM.append(running_costs_RANDOM[-1] + BATCH_COST)
            y_candidate_RANDOM = np.delete(y_candidate_RANDOM, indices_random, axis=0)
            costs_RANDOM = np.delete(costs_RANDOM, indices_random, axis=0)

        print(i, y_best_BO, "at Ntrain = {}".format(len(X)) , y_best_RANDOM)


    y_better_BO_ALL.append(y_better_BO)
    y_better_RANDOM_ALL.append(y_better_RANDOM)
    running_costs_BO_ALL.append(running_costs_BO)
    running_costs_RANDOM_ALL.append(running_costs_RANDOM)


y_better_BO_ALL = np.array(y_better_BO_ALL)
y_better_RANDOM_ALL = np.array(y_better_RANDOM_ALL)
running_costs_BO_ALL = np.array(running_costs_BO_ALL)
running_costs_RANDOM_ALL = np.array(running_costs_RANDOM_ALL)

plot_utility_BO_vs_RS(y_better_BO_ALL, y_better_RANDOM_ALL)
plot_costs_BO_vs_RS(running_costs_BO_ALL, running_costs_RANDOM_ALL)
import torch
import pdb
import numpy as np
import random
import copy as cp
from exp_configs_2 import *
from BO import *
from utils import *


def compute_price_acquisition(NEW_LIGANDS, price_dict):
    NEW_LIGANDS = np.unique(NEW_LIGANDS)
    price_acquisition = 0
    for ligand in NEW_LIGANDS:
        price_acquisition += price_dict[ligand]
    return price_acquisition


def update_price_dict(price_dict, NEW_LIGANDS):
    NEW_LIGANDS = np.unique(NEW_LIGANDS)
    for ligand in NEW_LIGANDS:
        price_dict[ligand] = 0
    return price_dict


RESULTS = []

for exp_config in benchmark:
    DATASET = Evaluation_data(exp_config["dataset"], exp_config["ntrain"], exp_config["prices"], init_strategy=exp_config["init_strategy"])
    bounds_norm = DATASET.bounds_norm

    N_RUNS            = exp_config["n_runs"]
    NITER             = exp_config["n_iter"]
    BATCH_SIZE        = exp_config["batch_size"]
    MAX_BATCH_COST    = exp_config["max_batch_cost"]
    COST_AWARE_BO     = exp_config["cost_aware"]
    COST_AWARE_RANDOM = False

    y_better_BO_ALL, y_better_RANDOM_ALL = [], []
    running_costs_BO_ALL, running_costs_RANDOM_ALL = [], []

    for run in range(N_RUNS):
        SEED = 111+run
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        X_init, y_init, costs_init, X_candidate, y_candidate, costs_candidate, LIGANDS_init, LIGANDS_candidate, price_dict =  DATASET.get_init_holdout_data(SEED)
        X, y = cp.deepcopy(X_init), cp.deepcopy(y_init)
        y_best = float(torch.max(y))
        model, scaler_y = update_model(X, y, bounds_norm)

        X_candidate_FULL, y_candidate_FULL = cp.deepcopy(X_candidate), cp.deepcopy(y_candidate)
        costs_FULL          = cp.deepcopy(costs_candidate)
        X_candidate_BO      = cp.deepcopy(X_candidate)
        y_candidate_BO      = cp.deepcopy(y_candidate)
        y_candidate_RANDOM  = cp.deepcopy(y_candidate).detach().numpy()

        running_costs_BO = [0]
        running_costs_RANDOM = [0]

        costs_BO        = cp.deepcopy(costs_candidate)
        costs_RANDOM    = cp.deepcopy(costs_candidate)

        price_dict_BO = cp.deepcopy(price_dict)
        price_dict_RANDOM = cp.deepcopy(price_dict)


        LIGANDS_candidate_BO = cp.deepcopy(LIGANDS_candidate)
        LIGANDS_candidate_RANDOM = cp.deepcopy(LIGANDS_candidate)


        y_better_BO = []
        y_better_RANDOM = []

        y_better_BO.append(y_best)
        y_better_RANDOM.append(y_best)
        y_best_BO, y_best_RANDOM = y_best, y_best


        for i in range(NITER):
            if COST_AWARE_BO == False:
                indices, candidates = gibbon_search(model, X_candidate_BO,bounds_norm, q=BATCH_SIZE,sequential=False, maximize=True)
                X, y = np.concatenate((X,candidates)), np.concatenate((y, y_candidate_BO[indices, :]))

                NEW_LIGANDS = LIGANDS_candidate_BO[indices]
                price_batch = compute_price_acquisition(NEW_LIGANDS, price_dict_BO)

                
                y_best_BO = check_better(y, y_best_BO)
                y_better_BO.append(y_best_BO)
                running_costs_BO.append((running_costs_BO[-1] + price_batch))
                model, _ = update_model(X, y, bounds_norm)
                X_candidate_BO = np.delete(X_candidate_BO, indices, axis=0)
                y_candidate_BO = np.delete(y_candidate_BO, indices, axis=0)
                LIGANDS_candidate_BO = np.delete(LIGANDS_candidate_BO, indices, axis=0)
                price_dict_BO = update_price_dict(price_dict_BO, NEW_LIGANDS)

            else:    
                indices, candidates = gibbon_search(model, X_candidate_BO,bounds_norm, q=2*BATCH_SIZE,sequential=False, maximize=True)
                NEW_LIGANDS = LIGANDS_candidate_BO[indices]
                #check if ligands are already in the batch and take the once with lowest in indices
                pdb.set_trace()
                
                suggested_costs = compute_price_acquisition(NEW_LIGANDS, price_dict)
                cheap_indices   = select_batch(suggested_costs, MAX_BATCH_COST, BATCH_SIZE)
                cheap_indices   = indices[cheap_indices]
                SUCCESS = True
                ITERATION = 1

                while (cheap_indices is None) or (len(cheap_indices) < BATCH_SIZE):
                    INCREMENTED_MAX_BATCH_COST = MAX_BATCH_COST
                    SUCCESS = False

                    INCREMENTED_BATCH_SIZE = BATCH_SIZE + ITERATION
                    print("Incrementing canditates for batch to: ", INCREMENTED_BATCH_SIZE)
                    if INCREMENTED_BATCH_SIZE > len(X_candidate_BO):
                        print("WTF")
                        #break
                        INCREMENTED_MAX_BATCH_COST  += 1

                    indices, candidates = gibbon_search(model, X_candidate_BO,bounds_norm, q=INCREMENTED_BATCH_SIZE,sequential=False, maximize=True)
                    suggested_costs = costs_BO[indices].flatten()
                    cheap_indices   = select_batch(suggested_costs, INCREMENTED_MAX_BATCH_COST, BATCH_SIZE)
                    cheap_indices   = indices[cheap_indices]

                    if cheap_indices is not None and len(cheap_indices) == BATCH_SIZE:
                        X, y = np.concatenate((X,X_candidate_BO[cheap_indices])), np.concatenate((y, y_candidate_BO[cheap_indices, :]))
                        y_best_BO = check_better(y, y_best_BO)

                        y_better_BO.append(y_best_BO)
                        BATCH_COST = sum(costs_BO[cheap_indices])[0]
                        print("Batch cost: ", BATCH_COST)
                        running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
                        model, scaler_y = update_model(X, y, bounds_norm)
                    

                        X_candidate_BO = np.delete(X_candidate_BO, cheap_indices, axis=0)
                        y_candidate_BO = np.delete(y_candidate_BO, cheap_indices, axis=0)
                        costs_BO = np.delete(costs_BO, cheap_indices, axis=0)
                        break
                    
                    ITERATION +=1

                if SUCCESS:
                    X, y = np.concatenate((X,X_candidate_BO[cheap_indices])), np.concatenate((y, y_candidate_BO[cheap_indices, :]))
                    y_best_BO = check_better(y, y_best_BO)
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
                NEW_LIGANDS = LIGANDS_candidate_RANDOM[indices_random]
                price_batch = compute_price_acquisition(NEW_LIGANDS, price_dict_RANDOM)



                if max(y_candidate_RANDOM[indices_random])[0] > y_best_RANDOM:
                    y_best_RANDOM = max(y_candidate_RANDOM[indices_random])[0] 

                y_better_RANDOM.append(y_best_RANDOM)
                
                running_costs_RANDOM.append(running_costs_RANDOM[-1] + price_batch)
                y_candidate_RANDOM = np.delete(y_candidate_RANDOM, indices_random, axis=0)
                LIGANDS_candidate_RANDOM = np.delete(LIGANDS_candidate_RANDOM, indices_random, axis=0)
                price_dict_RANDOM = update_price_dict(price_dict_RANDOM, NEW_LIGANDS)
            else:
                all_cheapest_indices = np.argwhere(costs_RANDOM.flatten() == 0).flatten()
                indices_random = np.random.choice(all_cheapest_indices, size=BATCH_SIZE, replace=False)
                NEW_LIGANDS = LIGANDS_candidate_RANDOM[indices_random]
                price_batch = compute_price_acquisition(NEW_LIGANDS, price_dict_RANDOM)



                if max(y_candidate_RANDOM[indices_random])[0] > y_best_RANDOM:
                    y_best_RANDOM = max(y_candidate_RANDOM[indices_random])[0]

                y_better_RANDOM.append(y_best_RANDOM)
                BATCH_COST = sum(costs_RANDOM[indices_random])[0]
                running_costs_RANDOM.append(running_costs_RANDOM[-1] + price_batch)
                y_candidate_RANDOM = np.delete(y_candidate_RANDOM, indices_random, axis=0)
                price_dict_RANDOM = update_price_dict(price_dict_RANDOM, NEW_LIGANDS)


            print("--------------------")
            print("# |{}/{}|\tBO {:.2f}\tRS {:.2f}\tSUM(COSTS BO): ${}\tN_train {}".format(i+1,NITER ,y_best_BO, y_best_RANDOM,running_costs_BO[-1],len(X)))

        y_better_BO_ALL.append(y_better_BO)
        y_better_RANDOM_ALL.append(y_better_RANDOM)
        running_costs_BO_ALL.append(running_costs_BO)
        running_costs_RANDOM_ALL.append(running_costs_RANDOM)


    y_better_BO_ALL = np.array(y_better_BO_ALL)
    y_better_RANDOM_ALL = np.array(y_better_RANDOM_ALL)

    max_n = reaching_max_n(y_better_BO_ALL)
    plot_utility_BO_vs_RS(y_better_BO_ALL, y_better_RANDOM_ALL, name="./figures/utility_{}_{}.png".format(exp_config["dataset"], exp_config["max_batch_cost"]))
    plot_costs_BO_vs_RS(running_costs_BO_ALL, running_costs_RANDOM_ALL, name="./figures/optimization_{}_{}.png".format(exp_config["dataset"], exp_config["max_batch_cost"]))

    RESULTS.append({"settings": exp_config,"max_n":max_n ,"y_better_BO_ALL": y_better_BO_ALL, 
                    "y_better_RANDOM_ALL": y_better_RANDOM_ALL, "running_costs_BO_ALL": running_costs_BO_ALL, 
                    "running_costs_RANDOM_ALL": running_costs_RANDOM_ALL})
    
    print("Done with experiment: ", exp_config)

print("Done with all experiments")
print("Saving results")
save_pkl(RESULTS, "results.pkl", )
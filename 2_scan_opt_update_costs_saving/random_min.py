import torch
import numpy as np
import random
import copy as cp
from BO import *
from utils import *
from experiments import *
import pdb


benchmark = [
    {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": True,
        "n_runs": 100, #5
        "n_iter": 15,
        "batch_size": 5,
        "max_batch_cost": 100.0,
        "ntrain": 200,
        "prices": "update_ligand_when_used",
    }
]


SEED = 111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if __name__ == "__main__":
    print("Starting experiments")
    # Modify such that maximal cost gets doubled every iteration
    RESULTS = []

    for exp_config in benchmark:
        print("Starting experiment: ", exp_config)
        y_better_RANDOM_ALL = []
        running_costs_RANDOM_ALL = []

        DATASET = Evaluation_data(
            exp_config["dataset"],
            exp_config["ntrain"],
            exp_config["prices"],
            init_strategy=exp_config["init_strategy"],
        )
        bounds_norm = DATASET.bounds_norm
        N_RUNS = exp_config["n_runs"]
        NITER = exp_config["n_iter"]
        MAX_BATCH_COST = exp_config["max_batch_cost"]
        BATCH_SIZE = exp_config["batch_size"]

        for run in range(N_RUNS):
            SEED = 111 + run
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)

            (
                X_init,
                y_init,
                costs_init,
                X_candidate,
                y_candidate,
                costs_candidate,
                LIGANDS_init,
                LIGANDS_candidate,
                price_dict,
            ) = DATASET.get_init_holdout_data(SEED)
            print(create_aligned_transposed_price_table(price_dict))
            X, y = cp.deepcopy(X_init), cp.deepcopy(y_init)
            y_best = float(torch.max(y))

            X_candidate_FULL, y_candidate_FULL = cp.deepcopy(X_candidate), cp.deepcopy(
                y_candidate
            )
            y_candidate_RANDOM = cp.deepcopy(y_candidate).detach().numpy()

            running_costs_RANDOM = [0]
            price_dict_RANDOM = cp.deepcopy(price_dict)

            LIGANDS_candidate_RANDOM = cp.deepcopy(LIGANDS_candidate)

            y_better_RANDOM = []

            y_better_RANDOM.append(y_best)
            y_best_RANDOM = y_best

            RANDOM_data = create_data_dict_RS_2A(
                y_candidate_RANDOM,
                y_best_RANDOM,
                LIGANDS_candidate_RANDOM,
                price_dict_RANDOM,
                BATCH_SIZE,
                MAX_BATCH_COST,
                y_better_RANDOM,
                running_costs_RANDOM,
            )

            for i in range(NITER):
                RANDOM_data = RS_STEP_2A(RANDOM_data)

                print("--------------------")
                print(
                    "# |{}/{}|RS {:.2f}  ${}\tSUM(COSTS RS): $".format(
                        i + 1,
                        NITER,
                        RANDOM_data["y_best_RANDOM"],
                        RANDOM_data["running_costs_RANDOM"][-1],
                    )
                )
                print(create_aligned_transposed_price_table(price_dict_RANDOM))

            y_better_RANDOM_ALL.append(RANDOM_data["y_better_RANDOM"])
            running_costs_RANDOM_ALL.append(RANDOM_data["running_costs_RANDOM"])

        y_better_RANDOM_ALL = np.array(y_better_RANDOM_ALL)

        RESULTS.append(
            {
                "settings": exp_config,
                "y_better_RANDOM_ALL": y_better_RANDOM_ALL,
                "running_costs_RANDOM_ALL": running_costs_RANDOM_ALL,
            }
        )

        print("Done with experiment: ", exp_config)

    print("Done with all experiments")
    print("Saving results")
    save_pkl(
        RESULTS,
        "results_random.pkl",
    )
    # pdb.set_trace()
    MEAN = np.mean(RESULTS[0]["y_better_RANDOM_ALL"], axis=0)
    print(MEAN)
    pdb.set_trace()
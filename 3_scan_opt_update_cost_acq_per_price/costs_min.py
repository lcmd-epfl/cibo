import torch
import numpy as np
import random
import copy as cp
from exp_configs_2 import *
from BO import *
from utils import *
from experiments import *


if __name__ == "__main__":
    print("Starting experiments")

    RESULTS = []

    for exp_config in benchmark:
        print("Starting experiment: ", exp_config)
        y_better_BO_ALL, y_better_RANDOM_ALL = [], []
        running_costs_BO_ALL, running_costs_RANDOM_ALL = [], []

        DATASET = Evaluation_data(
            exp_config["dataset"],
            exp_config["ntrain"],
            exp_config["prices"],
            init_strategy=exp_config["init_strategy"],
        )
        bounds_norm = DATASET.bounds_norm
        N_RUNS = exp_config["n_runs"]
        NITER = exp_config["n_iter"]
        BATCH_SIZE = exp_config["batch_size"]
        MAX_BATCH_COST = exp_config["max_batch_cost"]
        COST_AWARE_BO = exp_config["cost_aware"]

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
            model, scaler_y = update_model(X, y, bounds_norm)

            X_candidate_FULL, y_candidate_FULL = cp.deepcopy(X_candidate), cp.deepcopy(
                y_candidate
            )
            X_candidate_BO = cp.deepcopy(X_candidate)
            y_candidate_BO = cp.deepcopy(y_candidate)
            y_candidate_RANDOM = cp.deepcopy(y_candidate).detach().numpy()

            running_costs_BO = [0]
            running_costs_RANDOM = [0]

            price_dict_BO = cp.deepcopy(price_dict)
            price_dict_RANDOM = cp.deepcopy(price_dict)

            LIGANDS_candidate_BO = cp.deepcopy(LIGANDS_candidate)
            LIGANDS_candidate_RANDOM = cp.deepcopy(LIGANDS_candidate)

            y_better_BO = []
            y_better_RANDOM = []

            y_better_BO.append(y_best)
            y_better_RANDOM.append(y_best)
            y_best_BO, y_best_RANDOM = y_best, y_best

            BO_data = create_data_dict_BO_2A(
                model,
                y_best_BO,
                scaler_y,
                X,
                y,
                X_candidate_BO,
                y_candidate_BO,
                LIGANDS_candidate_BO,
                y_better_BO,
                price_dict_BO,
                running_costs_BO,
                bounds_norm,
                BATCH_SIZE,
                MAX_BATCH_COST,
            )

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
                if COST_AWARE_BO == False:
                    BO_data = BO_CASE_2A_STEP(BO_data)
                else:
                    BO_data = BO_AWARE_SCAN_FAST_CASE_2_STEP_ACQ_PRICE(BO_data)

                RANDOM_data = RS_STEP_2A(RANDOM_data)

                print("--------------------")
                print(
                    "# |{}/{}|\tBO {:.2f}\tRS {:.2f}\tSUM(COSTS BO): ${}\tSUM(COSTS RS): ${}\tN_train {}".format(
                        i + 1,
                        NITER,
                        BO_data["y_best_BO"],
                        RANDOM_data["y_best_RANDOM"],
                        BO_data["running_costs_BO"][-1],
                        RANDOM_data["running_costs_RANDOM"][-1],
                        BO_data["N_train"],
                    )
                )
                print(create_aligned_transposed_price_table(price_dict_BO))

            y_better_BO_ALL.append(BO_data["y_better_BO"])
            y_better_RANDOM_ALL.append(RANDOM_data["y_better_RANDOM"])
            running_costs_BO_ALL.append(BO_data["running_costs_BO"])
            running_costs_RANDOM_ALL.append(RANDOM_data["running_costs_RANDOM"])

        y_better_BO_ALL = np.array(y_better_BO_ALL)
        y_better_RANDOM_ALL = np.array(y_better_RANDOM_ALL)

        plot_utility_BO_vs_RS(
            y_better_BO_ALL,
            y_better_RANDOM_ALL,
            name="./figures/utility_{}_{}.png".format(
                exp_config["dataset"], exp_config["max_batch_cost"]
            ),
        )
        plot_costs_BO_vs_RS(
            running_costs_BO_ALL,
            running_costs_RANDOM_ALL,
            name="./figures/optimization_{}_{}.png".format(
                exp_config["dataset"], exp_config["max_batch_cost"]
            ),
        )

        RESULTS.append(
            {
                "settings": exp_config,
                "y_better_BO_ALL": y_better_BO_ALL,
                "y_better_RANDOM_ALL": y_better_RANDOM_ALL,
                "running_costs_BO_ALL": running_costs_BO_ALL,
                "running_costs_RANDOM_ALL": running_costs_RANDOM_ALL,
            }
        )

        print("Done with experiment: ", exp_config)

    print("Done with all experiments")
    print("Saving results")
    save_pkl(
        RESULTS,
        "results.pkl",
    )

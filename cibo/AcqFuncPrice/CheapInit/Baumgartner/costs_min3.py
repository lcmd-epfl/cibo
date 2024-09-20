import torch
import numpy as np
import random
import copy as cp
# from config import benchmark
from exp_configs_worst3 import benchmark


from cibo.BO import update_model
from cibo.utils import (
    plot_utility_BO_vs_RS,
    plot_costs_BO_vs_RS,
    create_aligned_transposed_price_table,
    data_dict_BO_LIGAND_BASE_SOLVENT,
    data_dict_RS_LIGAND_BASE_SOLVENT,
    save_pkl,
)
from cibo.experiments import (
    BO_LIGAND_BASE_SOLVENT,
    RS_LIGAND_BASE_SOLVENT,
    BO_COI_LIGAND_BASE_SOLVENT,
)
from cibo.data.datasets import Evaluation_data


SEED = 111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if __name__ == "__main__":
    print("Starting experiments")
    RESULTS = []

    for conf in benchmark:
        print("Starting experiment: ", conf)
        y_better_BO_ALL, y_better_RANDOM_ALL = [], []
        running_costs_BO_ALL, running_costs_RANDOM_ALL = [], []

        DATASET = Evaluation_data(
            conf["dataset"],
            conf["ntrain"],
            conf["prices"],
            init_strategy=conf["init_strategy"],
            nucleophile=conf["nucleophile"],
        )

        bounds_norm = DATASET.bounds_norm
        N_RUNS = conf["n_runs"]
        NITER = conf["n_iter"]
        BATCH_SIZE = conf["batch_size"]
        SURROGATE = conf["surrogate"]
        COST_AWARE_BO = conf["cost_aware"]

        for run in range(N_RUNS):
            SEED = 111 + run
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)

            (
                X_init,
                y_init,
                X_candidate,
                y_candidate,
                PRECATALYSTS_INIT,
                PRECATALYSTS_candidate,
                BASES_INIT,
                BASES_candidate,
                SOLVENTS_INIT,
                SOLVENTS_candidate,
                price_dict_precatalysts,
                price_dict_bases,
                price_dict_solvents,
                exp_init,
                exp_candidate,
            ) = DATASET.get_init_holdout_data(SEED)
            
            X, y = cp.deepcopy(X_init), cp.deepcopy(y_init)
            y_best = float(torch.max(y))
            model, scaler_y = update_model(X, y, bounds_norm, surrogate=SURROGATE)

            X_candidate_FULL, y_candidate_FULL = cp.deepcopy(X_candidate), cp.deepcopy(
                y_candidate
            )
            X_candidate_BO = cp.deepcopy(X_candidate)
            y_candidate_BO = cp.deepcopy(y_candidate)

            exp_candidate_BO = cp.deepcopy(exp_candidate)
            y_candidate_RANDOM = cp.deepcopy(y_candidate).detach().numpy()
            exp_candidate_RANDOM = cp.deepcopy(exp_candidate)

            running_costs_BO = [0]
            running_costs_RANDOM = [0]

            price_dict_BO_precatalysts = cp.deepcopy(price_dict_precatalysts)
            price_dict_RANDOM_precatalysts = cp.deepcopy(price_dict_precatalysts)

            price_dict_BO_bases = cp.deepcopy(price_dict_bases)
            price_dict_RANDOM_bases = cp.deepcopy(price_dict_bases)

            price_dict_BO_solvents = cp.deepcopy(price_dict_solvents)
            price_dict_RANDOM_solvents = cp.deepcopy(price_dict_solvents)

            PRECATALYSTS_candidate_BO = cp.deepcopy(PRECATALYSTS_candidate)
            PRECATALYSTS_candidate_RANDOM = cp.deepcopy(PRECATALYSTS_candidate)

            BASES_candidate_BO = cp.deepcopy(BASES_candidate)
            BASES_candidate_RANDOM = cp.deepcopy(BASES_candidate)

            SOLVENTS_candidate_BO = cp.deepcopy(SOLVENTS_candidate)
            SOLVENTS_candidate_RANDOM = cp.deepcopy(SOLVENTS_candidate)

            y_better_BO = []
            y_better_RANDOM = []

            y_better_BO.append(y_best)
            y_better_RANDOM.append(y_best)
            y_best_BO, y_best_RANDOM = y_best, y_best

            print("PRECATALYSTS")
            print(create_aligned_transposed_price_table(price_dict_BO_precatalysts))
            print("BASES")
            print(create_aligned_transposed_price_table(price_dict_BO_bases))
            print("SOLVENTS")
            print(create_aligned_transposed_price_table(price_dict_BO_solvents))

            BO_data = data_dict_BO_LIGAND_BASE_SOLVENT(
                model,
                y_best_BO,
                scaler_y,
                X,
                y,
                X_candidate_BO,
                y_candidate_BO,
                y_better_BO,
                price_dict_BO_precatalysts,
                price_dict_BO_bases,
                price_dict_BO_solvents,
                PRECATALYSTS_candidate_BO,
                BASES_candidate_BO,
                SOLVENTS_candidate_BO,
                running_costs_BO,
                bounds_norm,
                BATCH_SIZE,
                None,
                SURROGATE,
                conf["acq_func"],
                exp_init,
                exp_candidate_BO,
            )

            BO_data["cost_mod"] = conf["cost_mod"]

            RANDOM_data = data_dict_RS_LIGAND_BASE_SOLVENT(
                y_candidate_RANDOM,
                y_best_RANDOM,
                PRECATALYSTS_candidate_RANDOM,
                BASES_candidate_RANDOM,
                SOLVENTS_candidate_RANDOM,
                price_dict_RANDOM_precatalysts,
                price_dict_RANDOM_bases,
                price_dict_RANDOM_solvents,
                BATCH_SIZE,
                None,
                y_better_RANDOM,
                running_costs_RANDOM,
                exp_init,
                exp_candidate_RANDOM,
            )

            for i in range(NITER):
                if COST_AWARE_BO == False:
                    BO_data = BO_LIGAND_BASE_SOLVENT(BO_data)
                else:
                    BO_data = BO_COI_LIGAND_BASE_SOLVENT(BO_data)

                RANDOM_data = RS_LIGAND_BASE_SOLVENT(RANDOM_data)

                print("--------------------")

                print(
                    "# |{}/{}|\tBO {:.2f}\tRS {:.2f} \tSUM(COSTS BO): ${}\tSUM(COSTS RS): ${}\tN_train {}".format(
                        i + 1,
                        NITER,
                        BO_data["y_best_BO"],
                        RANDOM_data["y_best_RANDOM"],
                        BO_data["running_costs_BO"][-1],
                        RANDOM_data["running_costs_RANDOM"][-1],
                        BO_data["N_train"],
                    )
                )

            y_better_BO_ALL.append(BO_data["y_better_BO"])
            y_better_RANDOM_ALL.append(RANDOM_data["y_better_RANDOM"])
            running_costs_BO_ALL.append(BO_data["running_costs_BO"])
            running_costs_RANDOM_ALL.append(RANDOM_data["running_costs_RANDOM"])

        y_better_BO_ALL = np.array(y_better_BO_ALL)
        y_better_RANDOM_ALL = np.array(y_better_RANDOM_ALL)

        plot_utility_BO_vs_RS(
            y_better_BO_ALL,
            y_better_RANDOM_ALL,
            name="yield_{}_{}.png".format(
                conf["dataset"],
                conf["label"],
            ),
        )
        plot_costs_BO_vs_RS(
            running_costs_BO_ALL,
            running_costs_RANDOM_ALL,
            name="costs_{}_{}.png".format(
                conf["dataset"],
                conf["label"],
            ),
        )

        RESULTS.append(
            {
                "settings": conf,
                "y_better_BO_ALL": y_better_BO_ALL,
                "y_better_RANDOM_ALL": y_better_RANDOM_ALL,
                "running_costs_BO_ALL": running_costs_BO_ALL,
                "running_costs_RANDOM_ALL": running_costs_RANDOM_ALL,
                "exp_log": BO_data["EXP_DONE_BO"],
            }
        )

        print("Done with experiment: ", conf)

    print("Done with all experiments")
    print("Saving results")
    save_pkl(
        RESULTS,
        "cc_wof_results.pkl",
    )

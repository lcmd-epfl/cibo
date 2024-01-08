import numpy as np
import torch

from BO import (
    update_model,
    opt_gibbon,
    gibbon_search,
    opt_acqfct,
    opt_acqfct_cost,
    opt_acqfct_cost_B,
    opt_qNEI,
)

from utils import (
    check_better,
    update_X_y,
    compute_price_acquisition_ligands,
    update_price_dict_ligands,
    select_batch,
    check_success,
    find_optimal_batch,
)
import pdb


def RS_STEP(RANDOM_data):
    """
    Simple RS step without any cost constraints but keep track of the costs
    assuming random costs for each point and no cost updates.
    """
    # Extract the data from the dictionary
    y_candidate_RANDOM = RANDOM_data["y_candidate_RANDOM"]
    y_best_RANDOM = RANDOM_data["y_best_RANDOM"]
    costs_RANDOM = RANDOM_data["costs_RANDOM"]
    BATCH_SIZE = RANDOM_data["BATCH_SIZE"]
    y_better_RANDOM = RANDOM_data["y_better_RANDOM"]
    running_costs_RANDOM = RANDOM_data["running_costs_RANDOM"]

    # Select random indices for the batch
    indices_random = np.random.choice(
        np.arange(len(y_candidate_RANDOM)), size=BATCH_SIZE, replace=False
    )

    # Update the best value if a better one is found
    if max(y_candidate_RANDOM[indices_random])[0] > y_best_RANDOM:
        y_best_RANDOM = max(y_candidate_RANDOM[indices_random])[0]

    # Append the new best value and calculate batch cost
    y_better_RANDOM.append(y_best_RANDOM)
    BATCH_COST = sum(costs_RANDOM[indices_random])[0]
    running_costs_RANDOM.append(running_costs_RANDOM[-1] + BATCH_COST)

    # Update candidate and costs by removing selected indices
    y_candidate_RANDOM = np.delete(y_candidate_RANDOM, indices_random, axis=0)
    costs_RANDOM = np.delete(costs_RANDOM, indices_random, axis=0)

    # Update the dictionary with the new values
    RANDOM_data["y_candidate_RANDOM"] = y_candidate_RANDOM
    RANDOM_data["y_better_RANDOM"] = y_better_RANDOM
    RANDOM_data["y_best_RANDOM"] = y_best_RANDOM
    RANDOM_data["running_costs_RANDOM"] = running_costs_RANDOM
    RANDOM_data["costs_RANDOM"] = costs_RANDOM

    # BATCH_SIZE and MAX_BATCH_COST are constants and do not change

    # Return the updated dictionary
    return RANDOM_data


def BO_CASE_1_STEP(BO_data):
    """
    Simple BO step without any cost constraints but keep track of the costs
    """
    # Get current BO data from last iteration
    model = BO_data["model"]
    X, y = BO_data["X"], BO_data["y"]
    N_train = BO_data["N_train"]
    y_candidate_BO = BO_data["y_candidate_BO"]
    X_candidate_BO = BO_data["X_candidate_BO"]
    bounds_norm = BO_data["bounds_norm"]
    BATCH_SIZE = BO_data["BATCH_SIZE"]
    y_best_BO = BO_data["y_best_BO"]
    y_better_BO = BO_data["y_better_BO"]
    costs_BO = BO_data["costs_BO"]
    running_costs_BO = BO_data["running_costs_BO"]
    scaler_y = BO_data["scaler_y"]
    surrogate = BO_data["surrogate"]

    # Get new candidates
    indices, candidates = opt_gibbon(model, X_candidate_BO, bounds_norm, q=BATCH_SIZE)

    X, y = update_X_y(X, y, candidates, y_candidate_BO, indices)
    y_best_BO = check_better(y, y_best_BO)
    y_better_BO.append(y_best_BO)
    running_costs_BO.append((running_costs_BO[-1] + sum(costs_BO[indices]))[0])
    # Update model
    model, scaler_y = update_model(X, y, bounds_norm, surrogate=surrogate)
    # Delete candidates from pool of candidates since added to training data
    X_candidate_BO = np.delete(X_candidate_BO, indices, axis=0)
    y_candidate_BO = np.delete(y_candidate_BO, indices, axis=0)
    costs_BO = np.delete(costs_BO, indices, axis=0)

    # Update BO data for next iteration
    BO_data["model"] = model
    BO_data["X"], BO_data["y"] = X, y
    BO_data["N_train"] = len(X)
    BO_data["y_candidate_BO"] = y_candidate_BO
    BO_data["X_candidate_BO"] = X_candidate_BO
    BO_data["bounds_norm"] = bounds_norm
    BO_data["y_best_BO"] = y_best_BO
    BO_data["y_better_BO"] = y_better_BO
    BO_data["costs_BO"] = costs_BO
    BO_data["running_costs_BO"] = running_costs_BO
    BO_data["scaler_y"] = scaler_y

    return BO_data


def BO_AWARE_SCAN_FAST_CASE_1_STEP(BO_data):
    """
    BO with cost constraints on the costs per batch
    """
    model = BO_data["model"]
    X, y = BO_data["X"], BO_data["y"]
    N_train = BO_data["N_train"]
    y_candidate_BO = BO_data["y_candidate_BO"]
    X_candidate_BO = BO_data["X_candidate_BO"]
    bounds_norm = BO_data["bounds_norm"]
    BATCH_SIZE = BO_data["BATCH_SIZE"]
    y_best_BO = BO_data["y_best_BO"]
    y_better_BO = BO_data["y_better_BO"]
    costs_BO = BO_data["costs_BO"]
    running_costs_BO = BO_data["running_costs_BO"]
    scaler_y = BO_data["scaler_y"]
    MAX_BATCH_COST = BO_data["MAX_BATCH_COST"]
    acq_func = BO_data["acq_func"]
    surrogate = BO_data["surrogate"]

    index_set, _, _ = opt_acqfct(
        X,
        model,
        X_candidate_BO,
        bounds_norm,
        q=BATCH_SIZE,
        sequential=False,
        maximize=True,
        acq_func=acq_func,
    )
    SUCCESS = False
    for indices in index_set:
        suggested_costs = costs_BO[indices].flatten()

        if suggested_costs.sum() <= MAX_BATCH_COST:
            X, y = update_X_y(
                X,
                y,
                X_candidate_BO[indices],
                y_candidate_BO,
                indices,
            )
            y_best_BO = check_better(y, y_best_BO)

            BATCH_COST = sum(costs_BO[indices])[0]
            print("Batch cost1: ", BATCH_COST)
            running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
            model, scaler_y = update_model(X, y, bounds_norm, surrogate=surrogate)

            X_candidate_BO = np.delete(X_candidate_BO, indices, axis=0)
            y_candidate_BO = np.delete(y_candidate_BO, indices, axis=0)
            costs_BO = np.delete(costs_BO, indices, axis=0)
            SUCCESS = True
            break

    y_better_BO.append(y_best_BO)
    if not SUCCESS:
        BATCH_COST = 0
        print("Batch cost2: ", BATCH_COST)
        running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)

    # Update BO data for next iteration
    BO_data["model"] = model
    BO_data["X"], BO_data["y"] = X, y
    BO_data["N_train"] = len(X)
    BO_data["y_candidate_BO"] = y_candidate_BO
    BO_data["X_candidate_BO"] = X_candidate_BO
    BO_data["bounds_norm"] = bounds_norm
    BO_data["y_best_BO"] = y_best_BO
    BO_data["y_better_BO"] = y_better_BO
    BO_data["costs_BO"] = costs_BO
    BO_data["running_costs_BO"] = running_costs_BO
    BO_data["scaler_y"] = scaler_y

    return BO_data


def BO_AWARE_SCAN_CASE_1_STEP(BO_data):
    """
    BO with cost constraints on the costs per batch
    """
    model = BO_data["model"]
    X, y = BO_data["X"], BO_data["y"]
    N_train = BO_data["N_train"]
    y_candidate_BO = BO_data["y_candidate_BO"]
    X_candidate_BO = BO_data["X_candidate_BO"]
    bounds_norm = BO_data["bounds_norm"]
    BATCH_SIZE = BO_data["BATCH_SIZE"]
    y_best_BO = BO_data["y_best_BO"]
    y_better_BO = BO_data["y_better_BO"]
    costs_BO = BO_data["costs_BO"]
    running_costs_BO = BO_data["running_costs_BO"]
    scaler_y = BO_data["scaler_y"]
    MAX_BATCH_COST = BO_data["MAX_BATCH_COST"]
    surrogate = BO_data["surrogate"]

    SUCCESS = False
    ITERATION = 0

    while True:
        print(ITERATION)
        SUCCESS_1 = False
        indices, candidates = opt_gibbon(
            model,
            X_candidate_BO,
            bounds_norm,
            q=BATCH_SIZE,
            sequential=False,
            maximize=True,
            n_best=300,
            return_nr=ITERATION,
        )
        suggested_costs = costs_BO[indices].flatten()

        if suggested_costs.sum() <= MAX_BATCH_COST:
            X, y = update_X_y(
                X,
                y,
                X_candidate_BO[indices],
                y_candidate_BO,
                indices,
            )
            y_best_BO = check_better(y, y_best_BO)

            y_better_BO.append(y_best_BO)
            BATCH_COST = sum(costs_BO[indices])[0]
            print("Batch cost1: ", BATCH_COST)
            running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
            model, scaler_y = update_model(X, y, bounds_norm, surrogate=surrogate)

            X_candidate_BO = np.delete(X_candidate_BO, indices, axis=0)
            y_candidate_BO = np.delete(y_candidate_BO, indices, axis=0)
            costs_BO = np.delete(costs_BO, indices, axis=0)
            break

        ITERATION += 1

    # Update BO data for next iteration
    BO_data["model"] = model
    BO_data["X"], BO_data["y"] = X, y
    BO_data["N_train"] = len(X)
    BO_data["y_candidate_BO"] = y_candidate_BO
    BO_data["X_candidate_BO"] = X_candidate_BO
    BO_data["bounds_norm"] = bounds_norm
    BO_data["y_best_BO"] = y_best_BO
    BO_data["y_better_BO"] = y_better_BO
    BO_data["costs_BO"] = costs_BO
    BO_data["running_costs_BO"] = running_costs_BO
    BO_data["scaler_y"] = scaler_y

    return BO_data


def BO_AWARE_CASE_1_STEP(BO_data):
    """
    BO with cost constraints on the costs per batch
    """
    model = BO_data["model"]
    X, y = BO_data["X"], BO_data["y"]
    N_train = BO_data["N_train"]
    y_candidate_BO = BO_data["y_candidate_BO"]
    X_candidate_BO = BO_data["X_candidate_BO"]
    bounds_norm = BO_data["bounds_norm"]
    BATCH_SIZE = BO_data["BATCH_SIZE"]
    y_best_BO = BO_data["y_best_BO"]
    y_better_BO = BO_data["y_better_BO"]
    costs_BO = BO_data["costs_BO"]
    running_costs_BO = BO_data["running_costs_BO"]
    scaler_y = BO_data["scaler_y"]
    MAX_BATCH_COST = BO_data["MAX_BATCH_COST"]
    surrogate = BO_data["surrogate"]

    indices, _ = opt_gibbon(model, X_candidate_BO, bounds_norm, q=BATCH_SIZE)
    suggested_costs = costs_BO[indices].flatten()
    cheap_indices = select_batch(suggested_costs, MAX_BATCH_COST, BATCH_SIZE)
    cheap_indices, SUCCESS_1 = check_success(cheap_indices, indices)
    ITERATION = 1

    while len(cheap_indices) < BATCH_SIZE:
        INCREMENTED_MAX_BATCH_COST = MAX_BATCH_COST
        SUCCESS_1 = False

        INCREMENTED_BATCH_SIZE = BATCH_SIZE + ITERATION
        print("Incrementing canditates for batch to: ", INCREMENTED_BATCH_SIZE)
        if INCREMENTED_BATCH_SIZE > len(X_candidate_BO):
            print("Not enough candidates left to account for the costs")
            # therefore increasing the max batch cost to finally get enough candidates
            INCREMENTED_MAX_BATCH_COST += 1

        indices, _ = opt_gibbon(
            model, X_candidate_BO, bounds_norm, q=INCREMENTED_BATCH_SIZE
        )
        suggested_costs = costs_BO[indices].flatten()
        cheap_indices = select_batch(
            suggested_costs, INCREMENTED_MAX_BATCH_COST, BATCH_SIZE
        )
        cheap_indices, SUCCESS_2 = check_success(cheap_indices, indices)

        if len(cheap_indices) == BATCH_SIZE:
            X, y = update_X_y(
                X,
                y,
                X_candidate_BO[cheap_indices],
                y_candidate_BO,
                cheap_indices,
            )
            y_best_BO = check_better(y, y_best_BO)

            BATCH_COST = sum(costs_BO[cheap_indices])[0]
            print("Batch cost1: ", BATCH_COST)
            running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
            model, scaler_y = update_model(X, y, bounds_norm, surrogate=surrogate)

            X_candidate_BO = np.delete(X_candidate_BO, cheap_indices, axis=0)
            y_candidate_BO = np.delete(y_candidate_BO, cheap_indices, axis=0)
            costs_BO = np.delete(costs_BO, cheap_indices, axis=0)
            break

        ITERATION += 1

    if SUCCESS_1:
        X, y = update_X_y(
            X,
            y,
            X_candidate_BO[cheap_indices],
            y_candidate_BO,
            cheap_indices,
        )
        y_best_BO = check_better(y, y_best_BO)

        BATCH_COST = sum(costs_BO[cheap_indices])[0]
        print("Batch cost2: ", BATCH_COST)
        running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
        model, scaler_y = update_model(X, y, bounds_norm, surrogate=surrogate)
        X_candidate_BO = np.delete(X_candidate_BO, cheap_indices, axis=0)
        y_candidate_BO = np.delete(y_candidate_BO, cheap_indices, axis=0)
        costs_BO = np.delete(costs_BO, cheap_indices, axis=0)

    # Update BO data for next iteration
    y_better_BO.append(y_best_BO)
    if SUCCESS_1 == False and SUCCESS_2 == False:
        BATCH_COST = 0
        print("Batch cost3: ", BATCH_COST)
        running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)

    BO_data["model"] = model
    BO_data["X"], BO_data["y"] = X, y
    BO_data["N_train"] = len(X)
    BO_data["y_candidate_BO"] = y_candidate_BO
    BO_data["X_candidate_BO"] = X_candidate_BO
    BO_data["bounds_norm"] = bounds_norm
    BO_data["y_best_BO"] = y_best_BO
    BO_data["y_better_BO"] = y_better_BO
    BO_data["costs_BO"] = costs_BO
    BO_data["running_costs_BO"] = running_costs_BO
    BO_data["scaler_y"] = scaler_y

    return BO_data


def BO_CASE_2A_STEP(BO_data):
    """
    Normal BO with no cost constraints but keep track of the costs per batch for ca
    """
    # Get current BO data from last iteration
    model = BO_data["model"]
    X, y = BO_data["X"], BO_data["y"]
    N_train = BO_data["N_train"]
    y_candidate_BO = BO_data["y_candidate_BO"]
    X_candidate_BO = BO_data["X_candidate_BO"]
    bounds_norm = BO_data["bounds_norm"]
    BATCH_SIZE = BO_data["BATCH_SIZE"]
    y_best_BO = BO_data["y_best_BO"]
    y_better_BO = BO_data["y_better_BO"]
    LIGANDS_candidate_BO = BO_data["LIGANDS_candidate_BO"]
    price_dict_BO = BO_data["price_dict_BO"]
    running_costs_BO = BO_data["running_costs_BO"]
    scaler_y = BO_data["scaler_y"]
    surrogate = BO_data["surrogate"]
    acq_func = BO_data["acq_func"]

    # Assuming gibbon_search, update_X_y, compute_price_acquisition_ligands, check_better, update_model, and update_price_dict_ligands are defined elsewhere

    if acq_func == "NEI":
        indices, candidates = opt_qNEI(
            model,
            X_candidate_BO,
            bounds_norm,
            X,
            BATCH_SIZE,
            sequential=False,
            maximize=True,
        )

    elif acq_func == "GIBBON":
        indices, candidates = gibbon_search(
            model, X_candidate_BO, bounds_norm, q=BATCH_SIZE
        )
    else:
        raise NotImplementedError
    # TODO
    # pdb.set_trace()
    X, y = update_X_y(X, y, candidates, y_candidate_BO, indices)
    NEW_LIGANDS = LIGANDS_candidate_BO[indices]
    suggested_costs_all, _ = compute_price_acquisition_ligands(
        NEW_LIGANDS, price_dict_BO
    )
    y_best_BO = check_better(y, y_best_BO)
    y_better_BO.append(y_best_BO)
    running_costs_BO.append((running_costs_BO[-1] + suggested_costs_all))
    model, scaler_y = update_model(X, y, bounds_norm, surrogate=surrogate)
    X_candidate_BO = np.delete(X_candidate_BO, indices, axis=0)
    y_candidate_BO = np.delete(y_candidate_BO, indices, axis=0)
    LIGANDS_candidate_BO = np.delete(LIGANDS_candidate_BO, indices, axis=0)
    price_dict_BO = update_price_dict_ligands(price_dict_BO, NEW_LIGANDS)

    # Update BO data for next iteration
    BO_data["model"] = model
    BO_data["X"] = X
    BO_data["y"] = y
    BO_data["y_candidate_BO"] = y_candidate_BO
    BO_data["X_candidate_BO"] = X_candidate_BO
    BO_data["y_best_BO"] = y_best_BO
    BO_data["y_better_BO"] = y_better_BO
    BO_data["LIGANDS_candidate_BO"] = LIGANDS_candidate_BO
    BO_data["price_dict_BO"] = price_dict_BO
    BO_data["running_costs_BO"] = running_costs_BO
    BO_data["N_train"] = len(X)
    BO_data["scaler_y"] = scaler_y

    return BO_data


def BO_CASE_2B_STEP(BO_data):
    """
    Normal BO with no cost constraints but keep track of the costs per batch for ca
    """
    # Get current BO data from last iteration
    model = BO_data["model"]
    X, y = BO_data["X"], BO_data["y"]
    N_train = BO_data["N_train"]
    y_candidate_BO = BO_data["y_candidate_BO"]
    X_candidate_BO = BO_data["X_candidate_BO"]
    bounds_norm = BO_data["bounds_norm"]
    BATCH_SIZE = BO_data["BATCH_SIZE"]
    y_best_BO = BO_data["y_best_BO"]
    y_better_BO = BO_data["y_better_BO"]
    LIGANDS_candidate_BO = BO_data["LIGANDS_candidate_BO"]
    ADDITIVES_candidate_BO = BO_data["ADDITIVES_candidate_BO"]
    price_dict_BO_ligands = BO_data["price_dict_BO_ligands"]
    price_dict_BO_additives = BO_data["price_dict_BO_additives"]
    running_costs_BO = BO_data["running_costs_BO"]
    scaler_y = BO_data["scaler_y"]
    acq_func = BO_data["acq_func"]
    surrogate = BO_data["surrogate"]

    if acq_func == "NEI":
        indices, candidates = opt_qNEI(
            model,
            X_candidate_BO,
            bounds_norm,
            X,
            BATCH_SIZE,
            sequential=False,
            maximize=True,
        )

    elif acq_func == "GIBBON":
        indices, candidates = gibbon_search(
            model, X_candidate_BO, bounds_norm, q=BATCH_SIZE
        )
    else:
        raise NotImplementedError

    X, y = update_X_y(X, y, candidates, y_candidate_BO, indices)
    NEW_LIGANDS = LIGANDS_candidate_BO[indices]
    NEW_ADDITIVES = ADDITIVES_candidate_BO[indices]

    suggested_costs_all, price_per_ligand = compute_price_acquisition_ligands(
        NEW_LIGANDS, price_dict_BO_ligands
    )
    (
        suggested_costs_all_additives,
        price_per_additive,
    ) = compute_price_acquisition_ligands(NEW_ADDITIVES, price_dict_BO_additives)

    suggested_costs_all += suggested_costs_all_additives

    y_best_BO = check_better(y, y_best_BO)

    y_better_BO.append(y_best_BO)

    running_costs_BO.append((running_costs_BO[-1] + suggested_costs_all))

    model, scaler_y = update_model(X, y, bounds_norm, surrogate=surrogate)

    X_candidate_BO = np.delete(X_candidate_BO, indices, axis=0)
    y_candidate_BO = np.delete(y_candidate_BO, indices, axis=0)
    LIGANDS_candidate_BO = np.delete(LIGANDS_candidate_BO, indices, axis=0)
    ADDITIVES_candidate_BO = np.delete(ADDITIVES_candidate_BO, indices, axis=0)
    price_dict_BO_ligands = update_price_dict_ligands(
        price_dict_BO_ligands, NEW_LIGANDS
    )
    price_dict_BO_additives = update_price_dict_ligands(
        price_dict_BO_additives, NEW_ADDITIVES
    )

    # Update BO data for next iteration
    BO_data["model"] = model
    BO_data["X"] = X
    BO_data["y"] = y
    BO_data["y_candidate_BO"] = y_candidate_BO
    BO_data["X_candidate_BO"] = X_candidate_BO
    BO_data["y_best_BO"] = y_best_BO
    BO_data["y_better_BO"] = y_better_BO
    BO_data["LIGANDS_candidate_BO"] = LIGANDS_candidate_BO
    BO_data["ADDITIVES_candidate_BO"] = ADDITIVES_candidate_BO
    BO_data["price_dict_BO_ligands"] = price_dict_BO_ligands
    BO_data["price_dict_BO_additives"] = price_dict_BO_additives
    BO_data["running_costs_BO"] = running_costs_BO
    BO_data["N_train"] = len(X)
    BO_data["scaler_y"] = scaler_y

    return BO_data


def RS_STEP_2A(RANDOM_data):
    y_candidate_RANDOM = RANDOM_data["y_candidate_RANDOM"]
    BATCH_SIZE = RANDOM_data["BATCH_SIZE"]
    LIGANDS_candidate_RANDOM = RANDOM_data["LIGANDS_candidate_RANDOM"]
    price_dict_RANDOM = RANDOM_data["price_dict_RANDOM"]
    y_best_RANDOM = RANDOM_data["y_best_RANDOM"]
    y_better_RANDOM = RANDOM_data["y_better_RANDOM"]
    running_costs_RANDOM = RANDOM_data["running_costs_RANDOM"]

    indices_random = np.random.choice(
        np.arange(len(y_candidate_RANDOM)), size=BATCH_SIZE, replace=False
    )
    NEW_LIGANDS = LIGANDS_candidate_RANDOM[indices_random]
    suggested_costs_all, price_per_ligand = compute_price_acquisition_ligands(
        NEW_LIGANDS, price_dict_RANDOM
    )

    if max(y_candidate_RANDOM[indices_random])[0] > y_best_RANDOM:
        y_best_RANDOM = max(y_candidate_RANDOM[indices_random])[0]

    y_better_RANDOM.append(y_best_RANDOM)
    running_costs_RANDOM.append(running_costs_RANDOM[-1] + suggested_costs_all)

    y_candidate_RANDOM = np.delete(y_candidate_RANDOM, indices_random, axis=0)
    LIGANDS_candidate_RANDOM = np.delete(
        LIGANDS_candidate_RANDOM, indices_random, axis=0
    )
    price_dict_RANDOM = update_price_dict_ligands(price_dict_RANDOM, NEW_LIGANDS)

    # Update all modified quantities and return RANDOM_data
    RANDOM_data["y_candidate_RANDOM"] = y_candidate_RANDOM
    RANDOM_data["LIGANDS_candidate_RANDOM"] = LIGANDS_candidate_RANDOM
    RANDOM_data["price_dict_RANDOM"] = price_dict_RANDOM
    RANDOM_data["y_best_RANDOM"] = y_best_RANDOM
    RANDOM_data["y_better_RANDOM"] = y_better_RANDOM
    RANDOM_data["running_costs_RANDOM"] = running_costs_RANDOM

    return RANDOM_data


def RS_STEP_2B(RANDOM_data):
    y_candidate_RANDOM = RANDOM_data["y_candidate_RANDOM"]
    BATCH_SIZE = RANDOM_data["BATCH_SIZE"]
    LIGANDS_candidate_RANDOM = RANDOM_data["LIGANDS_candidate_RANDOM"]
    ADDITIVES_candidate_RANDOM = RANDOM_data["ADDITIVES_candidate_RANDOM"]
    price_dict_RANDOM_ligands = RANDOM_data["price_dict_RANDOM_ligands"]
    price_dict_RANDOM_additives = RANDOM_data["price_dict_RANDOM_additives"]
    y_best_RANDOM = RANDOM_data["y_best_RANDOM"]
    y_better_RANDOM = RANDOM_data["y_better_RANDOM"]
    running_costs_RANDOM = RANDOM_data["running_costs_RANDOM"]

    indices_random = np.random.choice(
        np.arange(len(y_candidate_RANDOM)), size=BATCH_SIZE, replace=False
    )

    NEW_LIGANDS = LIGANDS_candidate_RANDOM[indices_random]
    NEW_ADDITIVES = ADDITIVES_candidate_RANDOM[indices_random]

    suggested_costs_all, price_per_ligand = compute_price_acquisition_ligands(
        NEW_LIGANDS, price_dict_RANDOM_ligands
    )
    (
        suggested_costs_all_additives,
        price_per_additive,
    ) = compute_price_acquisition_ligands(NEW_ADDITIVES, price_dict_RANDOM_additives)

    suggested_costs_all += suggested_costs_all_additives

    if max(y_candidate_RANDOM[indices_random])[0] > y_best_RANDOM:
        y_best_RANDOM = max(y_candidate_RANDOM[indices_random])[0]

    y_better_RANDOM.append(y_best_RANDOM)
    running_costs_RANDOM.append(running_costs_RANDOM[-1] + suggested_costs_all)

    y_candidate_RANDOM = np.delete(y_candidate_RANDOM, indices_random, axis=0)
    LIGANDS_candidate_RANDOM = np.delete(
        LIGANDS_candidate_RANDOM, indices_random, axis=0
    )
    ADDITIVES_candidate_RANDOM = np.delete(
        ADDITIVES_candidate_RANDOM, indices_random, axis=0
    )

    price_dict_RANDOM_ligands = update_price_dict_ligands(
        price_dict_RANDOM_ligands, NEW_LIGANDS
    )
    price_dict_RANDOM_additives = update_price_dict_ligands(
        price_dict_RANDOM_additives, NEW_ADDITIVES
    )

    # Update all modified quantities and return RANDOM_data
    RANDOM_data["y_candidate_RANDOM"] = y_candidate_RANDOM
    RANDOM_data["LIGANDS_candidate_RANDOM"] = LIGANDS_candidate_RANDOM
    RANDOM_data["ADDITIVES_candidate_RANDOM"] = ADDITIVES_candidate_RANDOM
    RANDOM_data["price_dict_RANDOM_ligands"] = price_dict_RANDOM_ligands
    RANDOM_data["price_dict_RANDOM_additives"] = price_dict_RANDOM_additives
    RANDOM_data["y_best_RANDOM"] = y_best_RANDOM
    RANDOM_data["y_better_RANDOM"] = y_better_RANDOM
    RANDOM_data["running_costs_RANDOM"] = running_costs_RANDOM

    return RANDOM_data


def BO_AWARE_CASE_2A_STEP(BO_data):
    """
    BO with cost constraints on the costs per batch, updating prices when ligand first bought
    """

    # Get current BO data from last iteration
    model = BO_data["model"]
    X, y = BO_data["X"], BO_data["y"]
    N_train = BO_data["N_train"]
    y_candidate_BO = BO_data["y_candidate_BO"]
    X_candidate_BO = BO_data["X_candidate_BO"]
    bounds_norm = BO_data["bounds_norm"]
    BATCH_SIZE = BO_data["BATCH_SIZE"]
    y_best_BO = BO_data["y_best_BO"]
    y_better_BO = BO_data["y_better_BO"]
    LIGANDS_candidate_BO = BO_data["LIGANDS_candidate_BO"]
    price_dict_BO = BO_data["price_dict_BO"]
    running_costs_BO = BO_data["running_costs_BO"]
    MAX_BATCH_COST = BO_data["MAX_BATCH_COST"]
    scaler_y = BO_data["scaler_y"]
    surrogate = BO_data["surrogate"]
    acq_func = BO_data["acq_func"]

    SUCCESS_1 = False
    indices, candidates = gibbon_search(
        model, X_candidate_BO, bounds_norm, q=BATCH_SIZE
    )
    NEW_LIGANDS = LIGANDS_candidate_BO[indices]

    indices, SUCCESS_1, BATCH_COST, NEW_LIGANDS = find_optimal_batch(
        price_dict_BO, NEW_LIGANDS, indices, BATCH_SIZE, MAX_BATCH_COST
    )

    ITERATION = 1

    while SUCCESS_1 == False:
        INCREMENTED_MAX_BATCH_COST = MAX_BATCH_COST
        SUCCESS_1 = False

        INCREMENTED_BATCH_SIZE = BATCH_SIZE + ITERATION
        print(
            "Incrementing canditates for batch to: ",
            INCREMENTED_BATCH_SIZE,
        )
        if INCREMENTED_BATCH_SIZE > len(X_candidate_BO):
            print("Not enough candidates left to account for the costs")
            INCREMENTED_MAX_BATCH_COST += 1
        if INCREMENTED_BATCH_SIZE > 25:
            print("After 25 iterations, still cost conditions not met. Breaking...")
            INCREMENTED_MAX_BATCH_COST += 1
            break

        indices, candidates = opt_gibbon(
            model, X_candidate_BO, bounds_norm, q=INCREMENTED_BATCH_SIZE
        )
        NEW_LIGANDS = LIGANDS_candidate_BO[indices]
        print(indices)
        indices, SUCCESS_2, BATCH_COST, NEW_LIGANDS = find_optimal_batch(
            price_dict_BO, NEW_LIGANDS, indices, BATCH_SIZE, INCREMENTED_MAX_BATCH_COST
        )

        if SUCCESS_2:
            X, y = update_X_y(
                X,
                y,
                X_candidate_BO[indices],
                y_candidate_BO,
                indices,
            )
            y_best_BO = check_better(y, y_best_BO)

            print("Batch cost1: ", BATCH_COST)

            running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
            model, scaler_y = update_model(X, y, bounds_norm, surrogate=surrogate)
            X_candidate_BO = np.delete(X_candidate_BO, indices, axis=0)
            y_candidate_BO = np.delete(y_candidate_BO, indices, axis=0)
            LIGANDS_candidate_BO = np.delete(LIGANDS_candidate_BO, indices, axis=0)
            price_dict_BO = update_price_dict_ligands(price_dict_BO, NEW_LIGANDS)
            break

        ITERATION += 1

    if SUCCESS_1:
        X, y = update_X_y(
            X,
            y,
            X_candidate_BO[indices],
            y_candidate_BO,
            indices,
        )
        y_best_BO = check_better(y, y_best_BO)

        print("Batch cost2: ", BATCH_COST)
        running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
        model, scaler_y = update_model(X, y, bounds_norm, surrogate=surrogate)
        X_candidate_BO = np.delete(X_candidate_BO, indices, axis=0)
        y_candidate_BO = np.delete(y_candidate_BO, indices, axis=0)
        LIGANDS_candidate_BO = np.delete(LIGANDS_candidate_BO, indices, axis=0)
        price_dict_BO = update_price_dict_ligands(price_dict_BO, NEW_LIGANDS)

    # Update BO data for next iteration
    y_better_BO.append(y_best_BO)
    if not SUCCESS_1 and not SUCCESS_2:
        running_costs_BO.append(running_costs_BO[-1])

    BO_data["model"] = model
    BO_data["X"] = X
    BO_data["y"] = y
    BO_data["y_candidate_BO"] = y_candidate_BO
    BO_data["X_candidate_BO"] = X_candidate_BO
    BO_data["y_best_BO"] = y_best_BO
    BO_data["y_better_BO"] = y_better_BO
    BO_data["LIGANDS_candidate_BO"] = LIGANDS_candidate_BO
    BO_data["price_dict_BO"] = price_dict_BO
    BO_data["running_costs_BO"] = running_costs_BO
    BO_data["N_train"] = len(X)
    BO_data["scaler_y"] = scaler_y

    return BO_data


def BO_AWARE_SCAN_FAST_CASE_2_STEP(BO_data):
    model = BO_data["model"]
    X, y = BO_data["X"], BO_data["y"]
    N_train = BO_data["N_train"]
    y_candidate_BO = BO_data["y_candidate_BO"]
    X_candidate_BO = BO_data["X_candidate_BO"]
    bounds_norm = BO_data["bounds_norm"]
    BATCH_SIZE = BO_data["BATCH_SIZE"]
    y_best_BO = BO_data["y_best_BO"]
    y_better_BO = BO_data["y_better_BO"]
    LIGANDS_candidate_BO = BO_data["LIGANDS_candidate_BO"]
    price_dict_BO = BO_data["price_dict_BO"]
    running_costs_BO = BO_data["running_costs_BO"]
    MAX_BATCH_COST = BO_data["MAX_BATCH_COST"]
    scaler_y = BO_data["scaler_y"]
    acq_func = BO_data["acq_func"]
    surrogate = BO_data["surrogate"]

    index_set, _, _ = opt_acqfct(
        X,
        model,
        X_candidate_BO,
        bounds_norm,
        q=BATCH_SIZE,
        sequential=False,
        maximize=True,
        acq_func=acq_func,
    )

    price_list = np.array(list(price_dict_BO.values()))
    non_zero_prices = price_list[price_list > 0]
    if len(non_zero_prices) > 0:
        index_of_smallest_nonzero = np.where(price_list == non_zero_prices.min())[0][0]
        cheapest_ligand_price = price_list[index_of_smallest_nonzero]
        cheapest_ligand = list(price_dict_BO.keys())[index_of_smallest_nonzero]
        sorted_non_zero_prices = np.sort(non_zero_prices)
        if not len(sorted_non_zero_prices) > 1:
            print("Only one ligand left")

        if cheapest_ligand_price > MAX_BATCH_COST:
            print("No ligand can be bought with the current budget")
            print("Ask your boss for more $$$")

    else:
        print("All ligands have been bought")

    # select cheapest one that is not already 0 (correct that in the initialization)
    SUCCESS = False
    for indices in index_set:
        NEW_LIGANDS = LIGANDS_candidate_BO[indices]
        suggested_costs_all, _ = compute_price_acquisition_ligands(
            NEW_LIGANDS, price_dict_BO
        )

        BATCH_COST = suggested_costs_all
        if suggested_costs_all <= MAX_BATCH_COST:
            X, y = update_X_y(
                X,
                y,
                X_candidate_BO[indices],
                y_candidate_BO,
                indices,
            )
            y_best_BO = check_better(y, y_best_BO)
            y_better_BO.append(y_best_BO)

            print("Batch cost1: ", BATCH_COST)

            running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
            model, scaler_y = update_model(X, y, bounds_norm, surrogate=surrogate)
            X_candidate_BO = np.delete(X_candidate_BO, indices, axis=0)
            y_candidate_BO = np.delete(y_candidate_BO, indices, axis=0)
            LIGANDS_candidate_BO = np.delete(LIGANDS_candidate_BO, indices, axis=0)
            price_dict_BO = update_price_dict_ligands(price_dict_BO, NEW_LIGANDS)
            SUCCESS = True
            break

    if not SUCCESS:
        # means that only mixed ligand batches are suggested by the acqfct which we cannot afford,
        # thus take points from the cheapest ligand
        if cheapest_ligand_price < MAX_BATCH_COST:
            # find indices where LIGANDS_candidate_BO == cheapest_ligand
            NEW_LIGANDS = [cheapest_ligand]
            indices_cheap = np.where(LIGANDS_candidate_BO == cheapest_ligand)[0]

            index, _ = opt_gibbon(
                model, X_candidate_BO[indices_cheap], bounds_norm, q=5
            )

            X, y = update_X_y(
                X,
                y,
                X_candidate_BO[indices_cheap][index],
                y_candidate_BO,
                indices_cheap[index],
            )
            y_best_BO = check_better(y, y_best_BO)
            y_better_BO.append(y_best_BO)

            BATCH_COST = cheapest_ligand_price
            print("Batch cost2: ", BATCH_COST)
            running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
            model, scaler_y = update_model(X, y, bounds_norm, surrogate=surrogate)
            X_candidate_BO = np.delete(X_candidate_BO, indices_cheap[index], axis=0)
            y_candidate_BO = np.delete(y_candidate_BO, indices_cheap[index], axis=0)
            LIGANDS_candidate_BO = np.delete(
                LIGANDS_candidate_BO, indices_cheap[index], axis=0
            )
            price_dict_BO = update_price_dict_ligands(price_dict_BO, NEW_LIGANDS)
        else:
            index_of_zero = np.where(price_list == 0)[0][0]
            cheapest_ligand = list(price_dict_BO.keys())[index_of_zero]
            indices_cheap = np.where(LIGANDS_candidate_BO == cheapest_ligand)[0]
            if indices_cheap > 0:
                index, _ = opt_gibbon(
                    model, X_candidate_BO[indices_cheap], bounds_norm, q=5
                )
                X, y = update_X_y(
                    X,
                    y,
                    X_candidate_BO[indices_cheap][index],
                    y_candidate_BO,
                    indices_cheap[index],
                )
                y_best_BO = check_better(y, y_best_BO)
                y_better_BO.append(y_best_BO)

                BATCH_COST = 0
                print("Batch cost3: ", BATCH_COST)
                running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
                model, scaler_y = update_model(X, y, bounds_norm, surrogate=surrogate)
                X_candidate_BO = np.delete(X_candidate_BO, indices_cheap[index], axis=0)
                y_candidate_BO = np.delete(y_candidate_BO, indices_cheap[index], axis=0)
                LIGANDS_candidate_BO = np.delete(
                    LIGANDS_candidate_BO, indices_cheap[index], axis=0
                )
                price_dict_BO = update_price_dict_ligands(price_dict_BO, NEW_LIGANDS)
            else:
                print(
                    "All affordable ligands have been bought and no more free measurements possible. BO will stagnate now."
                )
                y_better_BO.append(y_best_BO)
                BATCH_COST = 0
                print("Batch cost3: ", BATCH_COST)
                running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)

    # Update BO data for next iteration
    BO_data["model"] = model
    BO_data["X"] = X
    BO_data["y"] = y
    BO_data["y_candidate_BO"] = y_candidate_BO
    BO_data["X_candidate_BO"] = X_candidate_BO
    BO_data["y_best_BO"] = y_best_BO
    BO_data["y_better_BO"] = y_better_BO
    BO_data["LIGANDS_candidate_BO"] = LIGANDS_candidate_BO
    BO_data["price_dict_BO"] = price_dict_BO
    BO_data["running_costs_BO"] = running_costs_BO
    BO_data["N_train"] = len(X)
    BO_data["scaler_y"] = scaler_y

    return BO_data


def BO_AWARE_SCAN_FAST_CASE_2_SAVED_BUDGET_STEP(BO_data):
    # Get current BO data from last iteration
    model = BO_data["model"]
    X, y = BO_data["X"], BO_data["y"]
    N_train = BO_data["N_train"]
    y_candidate_BO = BO_data["y_candidate_BO"]
    X_candidate_BO = BO_data["X_candidate_BO"]
    bounds_norm = BO_data["bounds_norm"]
    BATCH_SIZE = BO_data["BATCH_SIZE"]
    y_best_BO = BO_data["y_best_BO"]
    y_better_BO = BO_data["y_better_BO"]
    LIGANDS_candidate_BO = BO_data["LIGANDS_candidate_BO"]
    price_dict_BO = BO_data["price_dict_BO"]
    running_costs_BO = BO_data["running_costs_BO"]
    MAX_BATCH_COST = BO_data["MAX_BATCH_COST"]
    SAVED_BUDGET = BO_data["SAVED_BUDGET"]
    scaler_y = BO_data["scaler_y"]
    acq_func = BO_data["acq_func"]
    surrogate = BO_data["surrogate"]

    try:
        INCREASE_FACTOR = BO_data["INCREASE_FACTOR"]
    except:
        INCREASE_FACTOR = False

    if INCREASE_FACTOR:
        step_nr = BO_data["step_nr"]
        MAX_BATCH_COST *= step_nr

    index_set, _, _ = opt_acqfct(
        X,
        model,
        X_candidate_BO,
        bounds_norm,
        q=BATCH_SIZE,
        sequential=False,
        maximize=True,
        n_best=300,
        acq_func=acq_func,
    )

    price_list = np.array(list(price_dict_BO.values()))
    non_zero_prices = price_list[price_list > 0]
    if len(non_zero_prices) > 0:
        index_of_smallest_nonzero = np.where(price_list == non_zero_prices.min())[0][0]
        cheapest_ligand_price = price_list[index_of_smallest_nonzero]
        cheapest_ligand = list(price_dict_BO.keys())[index_of_smallest_nonzero]
        sorted_non_zero_prices = np.sort(non_zero_prices)
        if not len(sorted_non_zero_prices) > 1:
            print("Only one ligand left")

        if cheapest_ligand_price > SAVED_BUDGET:
            print("No ligand can be bought with the current budget")
            print("Ask your boss for more $$$")

    else:
        print("All ligands have been bought")

    # select cheapest one that is not already 0 (correct that in the initialization)
    SUCCESS = False
    for indices in index_set:
        NEW_LIGANDS = LIGANDS_candidate_BO[indices]
        suggested_costs_all, _ = compute_price_acquisition_ligands(
            NEW_LIGANDS, price_dict_BO
        )

        BATCH_COST = suggested_costs_all

        if suggested_costs_all <= SAVED_BUDGET:
            X, y = update_X_y(
                X,
                y,
                X_candidate_BO[indices],
                y_candidate_BO,
                indices,
            )
            y_best_BO = check_better(y, y_best_BO)
            y_better_BO.append(y_best_BO)

            print("Batch cost1: ", BATCH_COST)

            running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
            model, scaler_y = update_model(X, y, bounds_norm, surrogate=surrogate)
            X_candidate_BO = np.delete(X_candidate_BO, indices, axis=0)
            y_candidate_BO = np.delete(y_candidate_BO, indices, axis=0)
            LIGANDS_candidate_BO = np.delete(LIGANDS_candidate_BO, indices, axis=0)
            price_dict_BO = update_price_dict_ligands(price_dict_BO, NEW_LIGANDS)
            SUCCESS = True
            break

    if not SUCCESS:
        # means that only mixed ligand batches are suggested by the acqfct which we cannot afford,
        # thus take points from the cheapest ligand
        if cheapest_ligand_price < SAVED_BUDGET:
            # find indices where LIGANDS_candidate_BO == cheapest_ligand
            NEW_LIGANDS = [cheapest_ligand]
            indices_cheap = np.where(LIGANDS_candidate_BO == cheapest_ligand)[0]

            index, _ = opt_gibbon(
                model, X_candidate_BO[indices_cheap], bounds_norm, q=5
            )

            X, y = update_X_y(
                X,
                y,
                X_candidate_BO[indices_cheap][index],
                y_candidate_BO,
                indices_cheap[index],
            )
            y_best_BO = check_better(y, y_best_BO)
            y_better_BO.append(y_best_BO)

            BATCH_COST = cheapest_ligand_price
            print("Batch cost2: ", BATCH_COST)
            running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
            model, scaler_y = update_model(X, y, bounds_norm, surrogate=surrogate)
            X_candidate_BO = np.delete(X_candidate_BO, indices_cheap[index], axis=0)
            y_candidate_BO = np.delete(y_candidate_BO, indices_cheap[index], axis=0)
            LIGANDS_candidate_BO = np.delete(
                LIGANDS_candidate_BO, indices_cheap[index], axis=0
            )
            price_dict_BO = update_price_dict_ligands(price_dict_BO, NEW_LIGANDS)
        else:
            # if you cannot even afford the cheapest ligand, just do C or T measurements these come for free
            index_of_zero = np.where(price_list == 0)[0][0]
            cheapest_ligand = list(price_dict_BO.keys())[index_of_zero]
            indices_cheap = np.where(LIGANDS_candidate_BO == cheapest_ligand)[0]
            if indices_cheap > 0:
                index, _ = opt_gibbon(
                    model, X_candidate_BO[indices_cheap], bounds_norm, q=5
                )
                X, y = update_X_y(
                    X,
                    y,
                    X_candidate_BO[indices_cheap][index],
                    y_candidate_BO,
                    indices_cheap[index],
                )
                y_best_BO = check_better(y, y_best_BO)
                y_better_BO.append(y_best_BO)

                BATCH_COST = 0
                print("Batch cost3: ", BATCH_COST)
                running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
                model, scaler_y = update_model(X, y, bounds_norm, surrogate=surrogate)
                X_candidate_BO = np.delete(X_candidate_BO, indices_cheap[index], axis=0)
                y_candidate_BO = np.delete(y_candidate_BO, indices_cheap[index], axis=0)
                LIGANDS_candidate_BO = np.delete(
                    LIGANDS_candidate_BO, indices_cheap[index], axis=0
                )
                price_dict_BO = update_price_dict_ligands(price_dict_BO, NEW_LIGANDS)
            else:
                print(
                    "All affordable ligands have been bought and no more free measurements possible. So saving money for next iteration."
                )
                y_better_BO.append(y_best_BO)
                BATCH_COST = 0
                print("Batch cost3: ", BATCH_COST)
                running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)

    # Update BO data for next iteration
    SAVED_BUDGET = SAVED_BUDGET + (MAX_BATCH_COST - BATCH_COST)
    BO_data["SAVED_BUDGET"] = SAVED_BUDGET
    BO_data["model"] = model
    BO_data["X"] = X
    BO_data["y"] = y
    BO_data["y_candidate_BO"] = y_candidate_BO
    BO_data["X_candidate_BO"] = X_candidate_BO
    BO_data["y_best_BO"] = y_best_BO
    BO_data["y_better_BO"] = y_better_BO
    BO_data["LIGANDS_candidate_BO"] = LIGANDS_candidate_BO
    BO_data["price_dict_BO"] = price_dict_BO
    BO_data["running_costs_BO"] = running_costs_BO
    BO_data["N_train"] = len(X)
    BO_data["scaler_y"] = scaler_y

    return BO_data


def BO_AWARE_SCAN_FAST_CASE_2B_SAVED_BUDGET_STEP(BO_data):
    # Get current BO data from last iteration
    model = BO_data["model"]
    X, y = BO_data["X"], BO_data["y"]
    N_train = BO_data["N_train"]
    y_candidate_BO = BO_data["y_candidate_BO"]
    X_candidate_BO = BO_data["X_candidate_BO"]
    bounds_norm = BO_data["bounds_norm"]
    BATCH_SIZE = BO_data["BATCH_SIZE"]
    y_best_BO = BO_data["y_best_BO"]
    y_better_BO = BO_data["y_better_BO"]
    LIGANDS_candidate_BO = BO_data["LIGANDS_candidate_BO"]
    ADDITIVES_candidate_BO = BO_data["ADDITIVES_candidate_BO"]
    price_dict_BO_ligands = BO_data["price_dict_BO_ligands"]
    price_dict_BO_additives = BO_data["price_dict_BO_additives"]
    running_costs_BO = BO_data["running_costs_BO"]
    scaler_y = BO_data["scaler_y"]
    surrogate = BO_data["surrogate"]
    acq_func = BO_data["acq_func"]
    MAX_BATCH_COST = BO_data["MAX_BATCH_COST"]
    SAVED_BUDGET = BO_data["SAVED_BUDGET"]

    try:
        INCREASE_FACTOR = BO_data["INCREASE_FACTOR"]
    except:
        INCREASE_FACTOR = False

    if INCREASE_FACTOR:
        step_nr = BO_data["step_nr"]
        MAX_BATCH_COST *= step_nr

    index_set, _, _ = opt_acqfct(
        X,
        model,
        X_candidate_BO,
        bounds_norm,
        q=BATCH_SIZE,
        sequential=False,
        maximize=True,
        n_best=300,
        acq_func=acq_func,
    )

    price_list_ligands = np.array(list(price_dict_BO_ligands.values()))
    price_list_additives = np.array(list(price_dict_BO_additives.values()))

    non_zero_prices_ligands = price_list_ligands[price_list_ligands > 0]
    non_zero_prices_additives = price_list_additives[price_list_additives > 0]

    if len(non_zero_prices_ligands) > 0:
        index_of_smallest_nonzero_ligands = np.where(
            price_list_ligands == non_zero_prices_ligands.min()
        )[0][0]
        cheapest_ligand_price = price_list_ligands[index_of_smallest_nonzero_ligands]
        cheapest_ligand = list(price_dict_BO_ligands.keys())[
            index_of_smallest_nonzero_ligands
        ]
        sorted_non_zero_prices_ligands = np.sort(non_zero_prices_ligands)
        if not len(sorted_non_zero_prices_ligands) > 1:
            print("Only one ligand left")

        if cheapest_ligand_price > SAVED_BUDGET:
            print("No ligand can be bought with the current budget")
            print("Ask your boss for more $$$")

    else:
        print("All ligands have been bought")

    if len(non_zero_prices_additives) > 0:
        index_of_smallest_nonzero_additives = np.where(
            price_list_additives == non_zero_prices_additives.min()
        )[0][0]
        cheapest_additive_price = price_list_additives[
            index_of_smallest_nonzero_additives
        ]
        cheapest_additive = list(price_dict_BO_additives.keys())[
            index_of_smallest_nonzero_additives
        ]
        sorted_non_zero_prices_additives = np.sort(non_zero_prices_additives)
        if not len(sorted_non_zero_prices_additives) > 1:
            print("Only one additive left")

        if cheapest_additive_price > SAVED_BUDGET:
            print("No additive can be bought with the current budget")
            print("Ask your boss for more $$$")

    else:
        print("All additives have been bought")

    SUCCESS = False
    for indices in index_set:
        NEW_LIGANDS = LIGANDS_candidate_BO[indices]
        NEW_ADDITIVES = ADDITIVES_candidate_BO[indices]

        suggested_costs_all_ligands, _ = compute_price_acquisition_ligands(
            NEW_LIGANDS, price_dict_BO_ligands
        )
        suggested_costs_all_additives, _ = compute_price_acquisition_ligands(
            NEW_ADDITIVES, price_dict_BO_additives
        )

        BATCH_COST = suggested_costs_all_ligands + suggested_costs_all_additives
        print("Batch cost: ", BATCH_COST, "Saved budget: ", SAVED_BUDGET)
        if BATCH_COST <= SAVED_BUDGET:
            X, y = update_X_y(
                X,
                y,
                X_candidate_BO[indices],
                y_candidate_BO,
                indices,
            )
            y_best_BO = check_better(y, y_best_BO)
            y_better_BO.append(y_best_BO)

            print("Batch cost1: ", BATCH_COST)

            running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
            model, scaler_y = update_model(X, y, bounds_norm, surrogate=surrogate)
            X_candidate_BO = np.delete(X_candidate_BO, indices, axis=0)
            y_candidate_BO = np.delete(y_candidate_BO, indices, axis=0)
            LIGANDS_candidate_BO = np.delete(LIGANDS_candidate_BO, indices, axis=0)
            ADDITIVES_candidate_BO = np.delete(ADDITIVES_candidate_BO, indices, axis=0)
            price_dict_BO_ligands = update_price_dict_ligands(
                price_dict_BO_ligands, NEW_LIGANDS
            )
            price_dict_BO_additives = update_price_dict_ligands(
                price_dict_BO_additives, NEW_ADDITIVES
            )
            SUCCESS = True
            break

    if not SUCCESS:
        print(
            "Measure combinations of ligands and additives that were not explored but ingredients were bought"
        )

        index_of_zero_ligands = np.where(price_list_ligands == 0)[0]
        index_of_zero_additives = np.where(price_list_additives == 0)[0]

        free_ligands = np.array(list(price_dict_BO_ligands.keys()))[
            index_of_zero_ligands
        ]
        free_additives = np.array(list(price_dict_BO_additives.keys()))[
            index_of_zero_additives
        ]

        # find where free ligands in the candidates are
        free_ligand_indices = np.array(
            [
                i
                for i, ligand in enumerate(LIGANDS_candidate_BO)
                if ligand in free_ligands
            ]
        )

        # find where free additives in the candidates are
        free_additive_indices = np.array(
            [
                i
                for i, additive in enumerate(ADDITIVES_candidate_BO)
                if additive in free_additives
            ]
        )

        # find where both free ligands and additives in the candidates are
        free_ligand_additive_indices = np.intersect1d(
            free_ligand_indices, free_additive_indices
        )

        if len(free_ligand_additive_indices) >= BATCH_SIZE:
            suggested_costs_all_ligands, _ = compute_price_acquisition_ligands(
                LIGANDS_candidate_BO[free_ligand_additive_indices], price_dict_BO_ligands
            )

            suggested_costs_all_additives, _ = compute_price_acquisition_ligands(
                ADDITIVES_candidate_BO[free_ligand_additive_indices], price_dict_BO_additives
            )

            index, _ = opt_gibbon(
                model, X_candidate_BO[free_ligand_additive_indices], bounds_norm, q=5
            )
            y_best_BO = check_better(y, y_best_BO)
            y_better_BO.append(y_best_BO)

            BATCH_COST = 0
            print("Batch cost4: ", BATCH_COST)
            running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
            model, scaler_y = update_model(X, y, bounds_norm, surrogate=surrogate)
            X_candidate_BO = np.delete(
                X_candidate_BO, free_ligand_additive_indices[index], axis=0
            )
            y_candidate_BO = np.delete(
                y_candidate_BO, free_ligand_additive_indices[index], axis=0
            )
            LIGANDS_candidate_BO = np.delete(
                LIGANDS_candidate_BO, free_ligand_additive_indices[index], axis=0
            )
            ADDITIVES_candidate_BO = np.delete(
                ADDITIVES_candidate_BO, free_ligand_additive_indices[index], axis=0
            )

        else:
            print(
                "All affordable ligands have been bought and no more free measurements possible. So saving money for next iteration."
            )
            y_better_BO.append(y_best_BO)
            BATCH_COST = 0
            print("Batch cost5: ", BATCH_COST)
            running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)

    # Update BO data for next iteration
    SAVED_BUDGET = SAVED_BUDGET + (MAX_BATCH_COST - BATCH_COST)
    BO_data["SAVED_BUDGET"] = SAVED_BUDGET
    BO_data["model"] = model
    BO_data["X"] = X
    BO_data["y"] = y
    BO_data["y_candidate_BO"] = y_candidate_BO
    BO_data["X_candidate_BO"] = X_candidate_BO
    BO_data["y_best_BO"] = y_best_BO
    BO_data["y_better_BO"] = y_better_BO
    BO_data["LIGANDS_candidate_BO"] = LIGANDS_candidate_BO
    BO_data["ADDITIVES_candidate_BO"] = ADDITIVES_candidate_BO
    BO_data["price_dict_BO_ligands"] = price_dict_BO_ligands
    BO_data["price_dict_BO_additives"] = price_dict_BO_additives
    BO_data["running_costs_BO"] = running_costs_BO
    BO_data["N_train"] = len(X)
    BO_data["scaler_y"] = scaler_y

    return BO_data


def BO_AWARE_SCAN_FAST_CASE_2_STEP_ACQ_PRICE(BO_data):
    # Get current BO data from last iteration
    model = BO_data["model"]
    X, y = BO_data["X"], BO_data["y"]
    y_candidate_BO = BO_data["y_candidate_BO"]
    X_candidate_BO = BO_data["X_candidate_BO"]
    bounds_norm = BO_data["bounds_norm"]
    BATCH_SIZE = BO_data["BATCH_SIZE"]
    y_best_BO = BO_data["y_best_BO"]
    y_better_BO = BO_data["y_better_BO"]
    LIGANDS_candidate_BO = BO_data["LIGANDS_candidate_BO"]
    price_dict_BO = BO_data["price_dict_BO"]
    running_costs_BO = BO_data["running_costs_BO"]
    scaler_y = BO_data["scaler_y"]
    surrogate = BO_data["surrogate"]
    acq_func = BO_data["acq_func"]

    (
        index_set_rearranged,
        _,
        candidates_rearranged,
    ) = opt_acqfct_cost(
        X,
        model,
        X_candidate_BO,
        bounds_norm,
        q=BATCH_SIZE,
        LIGANDS_candidate_BO=LIGANDS_candidate_BO,
        price_dict_BO=price_dict_BO,
        acq_func=acq_func,
    )
    indices = index_set_rearranged[0]
    candidates = candidates_rearranged[0]
    # convert to back torch tensor
    candidates = torch.from_numpy(candidates).float()

    X, y = update_X_y(X, y, candidates, y_candidate_BO, indices)
    NEW_LIGANDS = LIGANDS_candidate_BO[indices]
    suggested_costs_all, _ = compute_price_acquisition_ligands(
        NEW_LIGANDS, price_dict_BO
    )
    y_best_BO = check_better(y, y_best_BO)
    y_better_BO.append(y_best_BO)
    running_costs_BO.append((running_costs_BO[-1] + suggested_costs_all))
    model, scaler_y = update_model(X, y, bounds_norm, surrogate=surrogate)
    X_candidate_BO = np.delete(X_candidate_BO, indices, axis=0)
    y_candidate_BO = np.delete(y_candidate_BO, indices, axis=0)
    LIGANDS_candidate_BO = np.delete(LIGANDS_candidate_BO, indices, axis=0)
    price_dict_BO = update_price_dict_ligands(price_dict_BO, NEW_LIGANDS)

    # Update BO data for next iteration
    BO_data["model"] = model
    BO_data["X"] = X
    BO_data["y"] = y
    BO_data["y_candidate_BO"] = y_candidate_BO
    BO_data["X_candidate_BO"] = X_candidate_BO
    BO_data["y_best_BO"] = y_best_BO
    BO_data["y_better_BO"] = y_better_BO
    BO_data["LIGANDS_candidate_BO"] = LIGANDS_candidate_BO
    BO_data["price_dict_BO"] = price_dict_BO
    BO_data["running_costs_BO"] = running_costs_BO
    BO_data["N_train"] = len(X)
    BO_data["scaler_y"] = scaler_y

    return BO_data


def BO_AWARE_SCAN_FAST_CASE_2B_STEP_ACQ_PRICE(BO_data):
    # Get current BO data from last iteration
    model = BO_data["model"]
    X, y = BO_data["X"], BO_data["y"]
    N_train = BO_data["N_train"]
    y_candidate_BO = BO_data["y_candidate_BO"]
    X_candidate_BO = BO_data["X_candidate_BO"]
    bounds_norm = BO_data["bounds_norm"]
    BATCH_SIZE = BO_data["BATCH_SIZE"]
    y_best_BO = BO_data["y_best_BO"]
    y_better_BO = BO_data["y_better_BO"]
    LIGANDS_candidate_BO = BO_data["LIGANDS_candidate_BO"]
    ADDITIVES_candidate_BO = BO_data["ADDITIVES_candidate_BO"]
    price_dict_BO_ligands = BO_data["price_dict_BO_ligands"]
    price_dict_BO_additives = BO_data["price_dict_BO_additives"]
    running_costs_BO = BO_data["running_costs_BO"]
    scaler_y = BO_data["scaler_y"]
    surrogate = BO_data["surrogate"]
    acq_func = BO_data["acq_func"]
    cost_mod = BO_data["cost_mod"]

    (
        index_set_rearranged,
        _,
        candidates_rearranged,
    ) = opt_acqfct_cost_B(
        X,
        model,
        X_candidate_BO,
        bounds_norm,
        q=BATCH_SIZE,
        LIGANDS_candidate_BO=LIGANDS_candidate_BO,
        ADDITIVES_candidate_BO=ADDITIVES_candidate_BO,
        price_dict_BO_ligands=price_dict_BO_ligands,
        price_dict_BO_additives=price_dict_BO_additives,
        acq_func=acq_func,
        cost_mod=cost_mod
    )

    indices = index_set_rearranged[0]
    candidates = candidates_rearranged[0]
    # convert to back torch tensor
    candidates = torch.from_numpy(candidates).float()

    X, y = update_X_y(X, y, candidates, y_candidate_BO, indices)

    NEW_LIGANDS = LIGANDS_candidate_BO[indices]
    NEW_ADDITIVES = ADDITIVES_candidate_BO[indices]

    suggested_costs_all, _ = compute_price_acquisition_ligands(
        NEW_LIGANDS, price_dict_BO_ligands
    )
    (
        suggested_costs_all_additives,
        _,
    ) = compute_price_acquisition_ligands(NEW_ADDITIVES, price_dict_BO_additives)

    suggested_costs_all += suggested_costs_all_additives

    y_best_BO = check_better(y, y_best_BO)

    y_better_BO.append(y_best_BO)

    running_costs_BO.append((running_costs_BO[-1] + suggested_costs_all))

    model, scaler_y = update_model(X, y, bounds_norm, surrogate=surrogate)

    X_candidate_BO = np.delete(X_candidate_BO, indices, axis=0)
    y_candidate_BO = np.delete(y_candidate_BO, indices, axis=0)
    LIGANDS_candidate_BO = np.delete(LIGANDS_candidate_BO, indices, axis=0)
    ADDITIVES_candidate_BO = np.delete(ADDITIVES_candidate_BO, indices, axis=0)
    price_dict_BO_ligands = update_price_dict_ligands(
        price_dict_BO_ligands, NEW_LIGANDS
    )
    price_dict_BO_additives = update_price_dict_ligands(
        price_dict_BO_additives, NEW_ADDITIVES
    )

    # Update BO data for next iteration
    BO_data["model"] = model
    BO_data["X"] = X
    BO_data["y"] = y
    BO_data["y_candidate_BO"] = y_candidate_BO
    BO_data["X_candidate_BO"] = X_candidate_BO
    BO_data["y_best_BO"] = y_best_BO
    BO_data["y_better_BO"] = y_better_BO
    BO_data["LIGANDS_candidate_BO"] = LIGANDS_candidate_BO
    BO_data["ADDITIVES_candidate_BO"] = ADDITIVES_candidate_BO
    BO_data["price_dict_BO_ligands"] = price_dict_BO_ligands
    BO_data["price_dict_BO_additives"] = price_dict_BO_additives
    BO_data["running_costs_BO"] = running_costs_BO
    BO_data["N_train"] = len(X)
    BO_data["scaler_y"] = scaler_y

    return BO_data

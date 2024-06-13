import numpy as np
import torch
from BO import (
    update_model,
    NEI_acqfct,
    GIBBON_acqfct,
    acqfct_COI_LIGAND,
    acqfct_COI_LIGAND_BASE_SOLVENT,
)

from utils import (
    check_better,
    update_X_y,
    compute_price_acquisition_ligands,
    update_price_dict_ligands,
)


def BO_LIGAND(BO_data):
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

    EXP_DONE_BO = BO_data["EXP_DONE_BO"]
    EXP_CANDIDATE_BO = BO_data["EXP_CANDIDATE_BO"]

    if acq_func == "NEI":
        indices, candidates = NEI_acqfct(
            model,
            X_candidate_BO,
            bounds_norm,
            X,
            BATCH_SIZE,
            sequential=False,
            maximize=True,
        )

    elif acq_func == "GIBBON":
        indices, candidates = GIBBON_acqfct(
            model, X_candidate_BO, bounds_norm, q=BATCH_SIZE
        )
    else:
        raise NotImplementedError

    X, y = update_X_y(X, y, candidates, y_candidate_BO, indices)
    NEW_LIGANDS = LIGANDS_candidate_BO[indices]
    suggested_costs_all, _ = compute_price_acquisition_ligands(
        NEW_LIGANDS, price_dict_BO
    )
    y_best_BO = check_better(y, y_best_BO)
    y_better_BO.append(y_best_BO)

    EXP_DONE_BO.append(EXP_CANDIDATE_BO[indices])
    EXP_CANDIDATE_BO = np.delete(EXP_CANDIDATE_BO, indices, axis=0)

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

    BO_data["EXP_DONE_BO"] = EXP_DONE_BO
    BO_data["EXP_CANDIDATE_BO"] = EXP_CANDIDATE_BO

    return BO_data


def BO_LIGAND_BASE_SOLVENT(BO_data):
    model = BO_data["model"]
    X, y = BO_data["X"], BO_data["y"]
    N_train = BO_data["N_train"]
    y_candidate_BO = BO_data["y_candidate_BO"]
    X_candidate_BO = BO_data["X_candidate_BO"]
    bounds_norm = BO_data["bounds_norm"]
    BATCH_SIZE = BO_data["BATCH_SIZE"]
    y_best_BO = BO_data["y_best_BO"]
    y_better_BO = BO_data["y_better_BO"]

    PRECATALYSTS_candidate_BO = BO_data["PRECATALYSTS_candidate_BO"]
    BASES_candidate_BO = BO_data["BASES_candidate_BO"]
    SOLVENTS_candidate_BO = BO_data["SOLVENTS_candidate_BO"]

    price_dict_BO_precatalysts = BO_data["price_dict_BO_precatalysts"]
    price_dict_BO_bases = BO_data["price_dict_BO_bases"]
    price_dict_BO_solvents = BO_data["price_dict_BO_solvents"]

    running_costs_BO = BO_data["running_costs_BO"]
    scaler_y = BO_data["scaler_y"]
    surrogate = BO_data["surrogate"]
    acq_func = BO_data["acq_func"]

    EXP_DONE_BO = BO_data["EXP_DONE_BO"]
    EXP_CANDIDATE_BO = BO_data["EXP_CANDIDATE_BO"]

    if acq_func == "NEI":
        indices, candidates = NEI_acqfct(
            model,
            X_candidate_BO,
            bounds_norm,
            X,
            BATCH_SIZE,
            sequential=False,
            maximize=True,
        )

    elif acq_func == "GIBBON":
        indices, candidates = GIBBON_acqfct(
            model, X_candidate_BO, bounds_norm, q=BATCH_SIZE
        )
    else:
        raise NotImplementedError

    X, y = update_X_y(X, y, candidates, y_candidate_BO, indices)

    NEW_PRECATALYSTS = PRECATALYSTS_candidate_BO[indices]
    NEW_BASES = BASES_candidate_BO[indices]
    NEW_SOLVENTS = SOLVENTS_candidate_BO[indices]

    suggested_costs_all, _ = compute_price_acquisition_ligands(
        NEW_PRECATALYSTS, price_dict_BO_precatalysts
    )

    suggested_costs_all_bases, _ = compute_price_acquisition_ligands(
        NEW_BASES, price_dict_BO_bases
    )

    suggested_costs_all_solvents, _ = compute_price_acquisition_ligands(
        NEW_SOLVENTS, price_dict_BO_solvents
    )

    suggested_costs_all += suggested_costs_all_bases + suggested_costs_all_solvents

    y_best_BO = check_better(y, y_best_BO)

    y_better_BO.append(y_best_BO)

    running_costs_BO.append((running_costs_BO[-1] + suggested_costs_all))

    model, scaler_y = update_model(X, y, bounds_norm, surrogate=surrogate)

    X_candidate_BO = np.delete(X_candidate_BO, indices, axis=0)
    y_candidate_BO = np.delete(y_candidate_BO, indices, axis=0)

    EXP_DONE_BO.append(EXP_CANDIDATE_BO[indices])
    EXP_CANDIDATE_BO = np.delete(EXP_CANDIDATE_BO, indices, axis=0)

    PRECATALYSTS_candidate_BO = np.delete(PRECATALYSTS_candidate_BO, indices, axis=0)
    BASES_candidate_BO = np.delete(BASES_candidate_BO, indices, axis=0)
    SOLVENTS_candidate_BO = np.delete(SOLVENTS_candidate_BO, indices, axis=0)

    price_dict_BO_precatalysts = update_price_dict_ligands(
        price_dict_BO_precatalysts, NEW_PRECATALYSTS
    )

    price_dict_BO_bases = update_price_dict_ligands(price_dict_BO_bases, NEW_BASES)

    price_dict_BO_solvents = update_price_dict_ligands(
        price_dict_BO_solvents, NEW_SOLVENTS
    )

    # Update BO data for next iteration
    BO_data["model"] = model
    BO_data["X"] = X
    BO_data["y"] = y
    BO_data["y_candidate_BO"] = y_candidate_BO
    BO_data["X_candidate_BO"] = X_candidate_BO
    BO_data["y_best_BO"] = y_best_BO
    BO_data["y_better_BO"] = y_better_BO
    BO_data["PRECATALYSTS_candidate_BO"] = PRECATALYSTS_candidate_BO
    BO_data["BASES_candidate_BO"] = BASES_candidate_BO
    BO_data["SOLVENTS_candidate_BO"] = SOLVENTS_candidate_BO
    BO_data["price_dict_BO_precatalysts"] = price_dict_BO_precatalysts
    BO_data["price_dict_BO_bases"] = price_dict_BO_bases
    BO_data["price_dict_BO_solvents"] = price_dict_BO_solvents
    BO_data["running_costs_BO"] = running_costs_BO
    BO_data["N_train"] = len(X)
    BO_data["scaler_y"] = scaler_y

    BO_data["EXP_DONE_BO"] = EXP_DONE_BO
    BO_data["EXP_CANDIDATE_BO"] = EXP_CANDIDATE_BO

    return BO_data


def RS_LIGAND(RANDOM_data):
    y_candidate_RANDOM = RANDOM_data["y_candidate_RANDOM"]
    BATCH_SIZE = RANDOM_data["BATCH_SIZE"]
    LIGANDS_candidate_RANDOM = RANDOM_data["LIGANDS_candidate_RANDOM"]
    price_dict_RANDOM = RANDOM_data["price_dict_RANDOM"]
    y_best_RANDOM = RANDOM_data["y_best_RANDOM"]
    y_better_RANDOM = RANDOM_data["y_better_RANDOM"]
    running_costs_RANDOM = RANDOM_data["running_costs_RANDOM"]
    EXP_DONE_RANDOM = RANDOM_data["EXP_DONE_RANDOM"]
    EXP_CANDIDATE_RANDOM = RANDOM_data["EXP_CANDIDATE_RANDOM"]

    indices_random = np.random.choice(
        np.arange(len(y_candidate_RANDOM)), size=BATCH_SIZE, replace=False
    )
    NEW_LIGANDS = LIGANDS_candidate_RANDOM[indices_random]
    suggested_costs_all, _ = compute_price_acquisition_ligands(
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

    EXP_DONE_RANDOM.append(EXP_CANDIDATE_RANDOM[indices_random])
    EXP_CANDIDATE_RANDOM = np.delete(EXP_CANDIDATE_RANDOM, indices_random, axis=0)

    # Update all modified quantities and return RANDOM_data
    RANDOM_data["y_candidate_RANDOM"] = y_candidate_RANDOM
    RANDOM_data["LIGANDS_candidate_RANDOM"] = LIGANDS_candidate_RANDOM
    RANDOM_data["price_dict_RANDOM"] = price_dict_RANDOM
    RANDOM_data["y_best_RANDOM"] = y_best_RANDOM
    RANDOM_data["y_better_RANDOM"] = y_better_RANDOM
    RANDOM_data["running_costs_RANDOM"] = running_costs_RANDOM

    RANDOM_data["EXP_DONE_RANDOM"] = EXP_DONE_RANDOM
    RANDOM_data["EXP_CANDIDATE_RANDOM"] = EXP_CANDIDATE_RANDOM

    return RANDOM_data


def RS_LIGAND_BASE_SOLVENT(RANDOM_data):
    y_candidate_RANDOM = RANDOM_data["y_candidate_RANDOM"]
    BATCH_SIZE = RANDOM_data["BATCH_SIZE"]
    PRECATALYSTS_candidate_RANDOM = RANDOM_data["PRECATALYSTS_candidate_RANDOM"]
    BASES_candidate_RANDOM = RANDOM_data["BASES_candidate_RANDOM"]
    SOLVENTS_candidate_RANDOM = RANDOM_data["SOLVENTS_candidate_RANDOM"]
    price_dict_RANDOM_precatalysts = RANDOM_data["price_dict_RANDOM_precatalysts"]
    price_dict_RANDOM_bases = RANDOM_data["price_dict_RANDOM_bases"]
    price_dict_RANDOM_solvents = RANDOM_data["price_dict_RANDOM_solvents"]

    y_best_RANDOM = RANDOM_data["y_best_RANDOM"]
    y_better_RANDOM = RANDOM_data["y_better_RANDOM"]

    running_costs_RANDOM = RANDOM_data["running_costs_RANDOM"]

    EXP_DONE_RANDOM = RANDOM_data["EXP_DONE_RANDOM"]
    EXP_CANDIDATE_RANDOM = RANDOM_data["EXP_CANDIDATE_RANDOM"]

    indices_random = np.random.choice(np.arange(len(y_candidate_RANDOM)), BATCH_SIZE)

    NEW_PRECATALYSTS = PRECATALYSTS_candidate_RANDOM[indices_random]
    NEW_BASES = BASES_candidate_RANDOM[indices_random]
    NEW_SOLVENTS = SOLVENTS_candidate_RANDOM[indices_random]

    suggested_costs_all, _ = compute_price_acquisition_ligands(
        NEW_PRECATALYSTS, price_dict_RANDOM_precatalysts
    )

    suggested_costs_all_bases, _ = compute_price_acquisition_ligands(
        NEW_BASES, price_dict_RANDOM_bases
    )

    suggested_costs_all_solvents, _ = compute_price_acquisition_ligands(
        NEW_SOLVENTS, price_dict_RANDOM_solvents
    )

    suggested_costs_all += suggested_costs_all_bases + suggested_costs_all_solvents

    if max(y_candidate_RANDOM[indices_random])[0] > y_best_RANDOM:
        y_best_RANDOM = max(y_candidate_RANDOM[indices_random])[0]

    y_better_RANDOM.append(y_best_RANDOM)

    EXP_DONE_RANDOM.append(EXP_CANDIDATE_RANDOM[indices_random])
    EXP_CANDIDATE_RANDOM = np.delete(EXP_CANDIDATE_RANDOM, indices_random, axis=0)

    running_costs_RANDOM.append((running_costs_RANDOM[-1] + suggested_costs_all))

    y_candidate_RANDOM = np.delete(y_candidate_RANDOM, indices_random, axis=0)
    PRECATALYSTS_candidate_RANDOM = np.delete(
        PRECATALYSTS_candidate_RANDOM, indices_random, axis=0
    )
    BASES_candidate_RANDOM = np.delete(BASES_candidate_RANDOM, indices_random, axis=0)
    SOLVENTS_candidate_RANDOM = np.delete(
        SOLVENTS_candidate_RANDOM, indices_random, axis=0
    )

    price_dict_RANDOM_precatalysts = update_price_dict_ligands(
        price_dict_RANDOM_precatalysts, NEW_PRECATALYSTS
    )

    price_dict_RANDOM_bases = update_price_dict_ligands(
        price_dict_RANDOM_bases, NEW_BASES
    )

    price_dict_RANDOM_solvents = update_price_dict_ligands(
        price_dict_RANDOM_solvents, NEW_SOLVENTS
    )

    RANDOM_data["y_candidate_RANDOM"] = y_candidate_RANDOM
    RANDOM_data["PRECATALYSTS_candidate_RANDOM"] = PRECATALYSTS_candidate_RANDOM
    RANDOM_data["BASES_candidate_RANDOM"] = BASES_candidate_RANDOM
    RANDOM_data["SOLVENTS_candidate_RANDOM"] = SOLVENTS_candidate_RANDOM
    RANDOM_data["price_dict_RANDOM_precatalysts"] = price_dict_RANDOM_precatalysts
    RANDOM_data["price_dict_RANDOM_bases"] = price_dict_RANDOM_bases
    RANDOM_data["price_dict_RANDOM_solvents"] = price_dict_RANDOM_solvents
    RANDOM_data["y_best_RANDOM"] = y_best_RANDOM
    RANDOM_data["y_better_RANDOM"] = y_better_RANDOM
    RANDOM_data["running_costs_RANDOM"] = running_costs_RANDOM
    RANDOM_data["EXP_DONE_RANDOM"] = EXP_DONE_RANDOM
    RANDOM_data["EXP_CANDIDATE_RANDOM"] = EXP_CANDIDATE_RANDOM

    return RANDOM_data


def BO_COI_LIGAND(BO_data):
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
    cost_mod = BO_data["cost_mod"]

    EXP_DONE_BO = BO_data["EXP_DONE_BO"]
    EXP_CANDIDATE_BO = BO_data["EXP_CANDIDATE_BO"]

    (
        index_set_rearranged,
        _,
        candidates_rearranged,
    ) = acqfct_COI_LIGAND(
        X,
        model,
        X_candidate_BO,
        bounds_norm,
        q=BATCH_SIZE,
        LIGANDS_candidate_BO=LIGANDS_candidate_BO,
        price_dict_BO=price_dict_BO,
        acq_func=acq_func,
        cost_mod=cost_mod,
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

    EXP_DONE_BO.append(EXP_CANDIDATE_BO[indices])
    EXP_CANDIDATE_BO = np.delete(EXP_CANDIDATE_BO, indices, axis=0)

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

    BO_data["EXP_DONE_BO"] = EXP_DONE_BO
    BO_data["EXP_CANDIDATE_BO"] = EXP_CANDIDATE_BO

    return BO_data


def BO_COI_LIGAND_BASE_SOLVENT(BO_data):
    model = BO_data["model"]
    X, y = BO_data["X"], BO_data["y"]
    N_train = BO_data["N_train"]
    y_candidate_BO = BO_data["y_candidate_BO"]
    X_candidate_BO = BO_data["X_candidate_BO"]
    bounds_norm = BO_data["bounds_norm"]
    BATCH_SIZE = BO_data["BATCH_SIZE"]
    y_best_BO = BO_data["y_best_BO"]
    y_better_BO = BO_data["y_better_BO"]

    PRECATALYSTS_candidate_BO = BO_data["PRECATALYSTS_candidate_BO"]
    BASES_candidate_BO = BO_data["BASES_candidate_BO"]
    SOLVENTS_candidate_BO = BO_data["SOLVENTS_candidate_BO"]

    price_dict_BO_precatalysts = BO_data["price_dict_BO_precatalysts"]
    price_dict_BO_bases = BO_data["price_dict_BO_bases"]
    price_dict_BO_solvents = BO_data["price_dict_BO_solvents"]

    running_costs_BO = BO_data["running_costs_BO"]
    scaler_y = BO_data["scaler_y"]
    surrogate = BO_data["surrogate"]
    acq_func = BO_data["acq_func"]
    cost_mod = BO_data["cost_mod"]

    EXP_DONE_BO = BO_data["EXP_DONE_BO"]
    EXP_CANDIDATE_BO = BO_data["EXP_CANDIDATE_BO"]


    (
        index_set_rearranged,
        _,
        candidates_rearranged,
    ) = acqfct_COI_LIGAND_BASE_SOLVENT(
        X,
        model,
        X_candidate_BO,
        bounds_norm,
        q=BATCH_SIZE,
        PRECATALYSTS_candidate_BO=PRECATALYSTS_candidate_BO,
        BASES_candidate_BO=BASES_candidate_BO,
        SOLVENTS_candidate_BO=SOLVENTS_candidate_BO,
        price_dict_BO_precatalysts=price_dict_BO_precatalysts,
        price_dict_BO_bases=price_dict_BO_bases,
        price_dict_BO_solvents=price_dict_BO_solvents,
        acq_func=acq_func,
        cost_mod=cost_mod,
    )

    indices = index_set_rearranged[0]
    candidates = candidates_rearranged[0]
    # convert to back torch tensor
    candidates = torch.from_numpy(candidates).float()
    X, y = update_X_y(X, y, candidates, y_candidate_BO, indices)
    NEW_PRECATALYSTS = PRECATALYSTS_candidate_BO[indices]
    NEW_BASES = BASES_candidate_BO[indices]
    NEW_SOLVENTS = SOLVENTS_candidate_BO[indices]

    suggested_costs_all, _ = compute_price_acquisition_ligands(
        NEW_PRECATALYSTS, price_dict_BO_precatalysts
    )

    suggested_costs_all_bases, _ = compute_price_acquisition_ligands(
        NEW_BASES, price_dict_BO_bases
    )

    suggested_costs_all_solvents, _ = compute_price_acquisition_ligands(
        NEW_SOLVENTS, price_dict_BO_solvents
    )

    suggested_costs_all += suggested_costs_all_bases + suggested_costs_all_solvents

    y_best_BO = check_better(y, y_best_BO)

    y_better_BO.append(y_best_BO)

    running_costs_BO.append((running_costs_BO[-1] + suggested_costs_all))

    model, scaler_y = update_model(X, y, bounds_norm, surrogate=surrogate)

    X_candidate_BO = np.delete(X_candidate_BO, indices, axis=0)
    y_candidate_BO = np.delete(y_candidate_BO, indices, axis=0)

    EXP_DONE_BO.append(EXP_CANDIDATE_BO[indices])
    EXP_CANDIDATE_BO = np.delete(EXP_CANDIDATE_BO, indices, axis=0)


    PRECATALYSTS_candidate_BO = np.delete(PRECATALYSTS_candidate_BO, indices, axis=0)
    BASES_candidate_BO = np.delete(BASES_candidate_BO, indices, axis=0)
    SOLVENTS_candidate_BO = np.delete(SOLVENTS_candidate_BO, indices, axis=0)

    price_dict_BO_precatalysts = update_price_dict_ligands(
        price_dict_BO_precatalysts, NEW_PRECATALYSTS
    )

    price_dict_BO_bases = update_price_dict_ligands(price_dict_BO_bases, NEW_BASES)

    price_dict_BO_solvents = update_price_dict_ligands(
        price_dict_BO_solvents, NEW_SOLVENTS
    )

    # Update BO data for next iteration
    BO_data["model"] = model
    BO_data["X"] = X
    BO_data["y"] = y
    BO_data["y_candidate_BO"] = y_candidate_BO
    BO_data["X_candidate_BO"] = X_candidate_BO
    BO_data["y_best_BO"] = y_best_BO
    BO_data["y_better_BO"] = y_better_BO
    BO_data["PRECATALYSTS_candidate_BO"] = PRECATALYSTS_candidate_BO
    BO_data["BASES_candidate_BO"] = BASES_candidate_BO
    BO_data["SOLVENTS_candidate_BO"] = SOLVENTS_candidate_BO
    BO_data["price_dict_BO_precatalysts"] = price_dict_BO_precatalysts
    BO_data["price_dict_BO_bases"] = price_dict_BO_bases
    BO_data["price_dict_BO_solvents"] = price_dict_BO_solvents
    BO_data["running_costs_BO"] = running_costs_BO
    BO_data["N_train"] = len(X)
    BO_data["scaler_y"] = scaler_y

    BO_data["EXP_DONE_BO"] = EXP_DONE_BO
    BO_data["EXP_CANDIDATE_BO"] = EXP_CANDIDATE_BO

    return BO_data

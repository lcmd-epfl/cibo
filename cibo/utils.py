# Standard library imports
import copy as cp
import os
import pickle

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import torch

# Specific module imports
from rdkit import Chem
from rdkit.Chem import AllChem

# savepkl file
def save_pkl(file, name):
    with open(name, "wb") as f:
        pickle.dump(file, f)


# load pkl file
def load_pkl(name):
    with open(name, "rb") as f:
        return pickle.load(f)


"""
Cheminformatics utility functions.
"""


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return None


def canonicalize_smiles_list(smiles_list):
    return [canonicalize_smiles(smiles) for smiles in smiles_list]


def inchi_to_smiles(inchi_list):
    """
    Convert a list of InChI strings to a list of canonical SMILES strings.

    Args:
    inchi_list (list): A list of InChI strings.

    Returns:
    list: A list of canonical SMILES strings.
    """
    smiles_list = []
    for inchi in inchi_list:
        mol = Chem.MolFromInchi(inchi)
        if mol:
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
        else:
            smiles_list.append(None)  # Append None for invalid InChI strings
    return smiles_list


class FingerprintGenerator:
    def __init__(self, nBits=512, radius=2):
        self.nBits = nBits
        self.radius = radius

    def featurize(self, smiles_list):
        fingerprints = []
        for smiles in smiles_list:
            if not isinstance(smiles, str):
                fingerprints.append(np.ones(self.nBits))
            else:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, self.radius, nBits=self.nBits
                    )
                    fp_array = np.array(
                        list(fp.ToBitString()), dtype=int
                    )  # Convert to NumPy array
                    fingerprints.append(fp_array)
                else:
                    pdb.set_trace()
                    print(f"Could not generate a molecule from SMILES: {smiles}")
                    fingerprints.append(np.array([None]))

        return np.array(fingerprints)


"""
Functions to manipulate arrays and tensors or convert types
"""


def convert2pytorch(X, y):
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float().reshape(-1, 1)
    return X, y


def check_entries(array_of_arrays):
    """
    Check if the entries of the arrays are between 0 and 1.
    Needed for for the datasets.py script.
    """

    for array in array_of_arrays:
        for item in array:
            if item < 0 or item > 1:
                return False
    return True


def check_better(y, y_best_BO):
    """
    Check if one of the molecuels in the new batch
    is better than the current best one.
    """

    if max(y)[0] > y_best_BO:
        return max(y)[0]
    else:
        return y_best_BO


def plot_utility_BO_vs_RS(y_better_BO_ALL, y_better_RANDOM_ALL, name="utility.png"):
    """
    Plot the utility of the BO vs RS (Random Search) for each iteration.
    """
    if not os.path.exists("./figures"):
        os.makedirs("./figures")

    y_BO_MEAN, y_BO_STD = np.mean(y_better_BO_ALL, axis=0), np.std(
        y_better_BO_ALL, axis=0
    )

    y_RANDOM_MEAN, y_RANDOM_STD = np.mean(y_better_RANDOM_ALL, axis=0), np.std(
        y_better_RANDOM_ALL, axis=0
    )

    lower_rnd = y_RANDOM_MEAN - y_RANDOM_STD
    upper_rnd = y_RANDOM_MEAN + y_RANDOM_STD
    lower_bo = y_BO_MEAN - y_BO_STD
    upper_bo = y_BO_MEAN + y_BO_STD

    NITER = len(y_BO_MEAN)
    fig1, ax1 = plt.subplots()

    ax1.plot(np.arange(NITER), y_RANDOM_MEAN, label="Random")
    ax1.fill_between(np.arange(NITER), lower_rnd, upper_rnd, alpha=0.2)
    ax1.plot(np.arange(NITER), y_BO_MEAN, label="Acquisition Function")
    ax1.fill_between(np.arange(NITER), lower_bo, upper_bo, alpha=0.2)
    ax1.set_xlabel("Number of Iterations")
    ax1.set_ylabel("Best Objective Value")
    plt.legend(loc="lower right")
    plt.xticks(list(np.arange(NITER)))

    # check if subfolder exists
    if not os.path.exists("./figures"):
        os.makedirs("./figures")

    plt.savefig("./figures/" + name)

    plt.clf()


def plot_costs_BO_vs_RS(
    running_costs_BO_ALL, running_costs_RANDOM_ALL, name="costs.png"
):
    """
    Plot the running costs of the BO vs RS (Random Search) for each iteration.
    """

    running_costs_BO_ALL_MEAN, running_costs_BO_ALL_STD = np.mean(
        running_costs_BO_ALL, axis=0
    ), np.std(running_costs_BO_ALL, axis=0)
    running_costs_RANDOM_ALL_MEAN, running_costs_RANDOM_ALL_STD = np.mean(
        running_costs_RANDOM_ALL, axis=0
    ), np.std(running_costs_RANDOM_ALL, axis=0)
    lower_rnd_costs = running_costs_RANDOM_ALL_MEAN - running_costs_RANDOM_ALL_STD
    upper_rnd_costs = running_costs_RANDOM_ALL_MEAN + running_costs_RANDOM_ALL_STD
    lower_bo_costs = running_costs_BO_ALL_MEAN - running_costs_BO_ALL_STD
    upper_bo_costs = running_costs_BO_ALL_MEAN + running_costs_BO_ALL_STD

    fig2, ax2 = plt.subplots()
    NITER = len(running_costs_BO_ALL_MEAN)

    ax2.plot(np.arange(NITER), running_costs_RANDOM_ALL_MEAN, label="Random")
    ax2.fill_between(np.arange(NITER), lower_rnd_costs, upper_rnd_costs, alpha=0.2)
    ax2.plot(np.arange(NITER), running_costs_BO_ALL_MEAN, label="Acquisition Function")
    ax2.fill_between(np.arange(NITER), lower_bo_costs, upper_bo_costs, alpha=0.2)
    ax2.set_xlabel("Number of Iterations")
    ax2.set_ylabel("Running Costs [$]")
    plt.legend(loc="lower right")
    plt.xticks(list(np.arange(NITER)))

    # check if subfolder exists
    if not os.path.exists("./figures"):
        os.makedirs("./figures")

    plt.savefig("./figures/" + name)
    plt.clf()


def check_success(cheap_indices, indices):
    if cheap_indices == []:
        return cheap_indices, False
    else:
        cheap_indices = indices[cheap_indices]
        return cheap_indices, True


def compute_price_acquisition_ligands(NEW_LIGANDS, price_dict):
    """
    This function is for 2_greedy_update_costs:
    Computes the price for the batch. When a ligand
    was first seen its price is added to the price_acquisition.
    If it is seen again in the same batch its price is 0.
    """

    price_acquisition = 0

    check_dict = cp.deepcopy(price_dict)
    for ind, ligand in enumerate(NEW_LIGANDS):
        price_acquisition += check_dict[ligand]
        check_dict[ligand] = 0

    check_dict = cp.deepcopy(price_dict)
    price_per_ligand = []
    for ligand in NEW_LIGANDS:
        price_per_ligand.append(check_dict[ligand])
        check_dict[ligand] = 0

    price_per_ligand = np.array(price_per_ligand)

    return price_acquisition, price_per_ligand


def function_cost_minus(NEW_LIGANDS, price_dict, acq_values):
    acq_values_max = max(cp.deepcopy(acq_values).flatten())
    check_dict_ligands = cp.deepcopy(price_dict)

    max_value = np.mean(np.array((list((check_dict_ligands.values())))))

    scaling_factor = acq_values_max / max_value

    for key in check_dict_ligands:
        check_dict_ligands[key] *= scaling_factor

    price_per_ligand = []
    for ligand in NEW_LIGANDS:
        minus = check_dict_ligands[ligand]
        price_per_ligand.append(minus)
        check_dict_ligands[ligand] = 0

    price_per_ligand = np.array(price_per_ligand)

    return price_per_ligand


def function_cost_C_minus(
    precatalysts,
    bases,
    solvents,
    price_dict_BO_precatalysts,
    price_dict_BO_bases,
    price_dict_BO_solvents,
    acq_values,
):
    """
    Only used for gibbon_search_modified_all_per_price!

    Computes the price for the batch. When a ligand is already in the inventory
    it will have a price of 1 to make sure the acfct value is not divided by 0.
    """
    acq_values_max = max(cp.deepcopy(acq_values).flatten())
    check_dict_precatalysts = cp.deepcopy(price_dict_BO_precatalysts)
    check_dict_bases = cp.deepcopy(price_dict_BO_bases)
    check_dict_solvents = cp.deepcopy(price_dict_BO_solvents)

    combined_dict = {
        **check_dict_precatalysts,
        **check_dict_bases,
        **check_dict_solvents,
    }

    # Find the maximum value in the combined dictionary
    max_combined_value = np.mean(np.array((list((combined_dict.values())))))

    # Calculate the scaling factor
    scaling_factor = acq_values_max / max_combined_value

    # Scale the values in check_dict_ligands
    for key in check_dict_precatalysts:
        check_dict_precatalysts[key] *= scaling_factor

    # Scale the values in check_dict_additives
    for key in check_dict_bases:
        check_dict_bases[key] *= scaling_factor

    # Scale the values in check_dict_additives
    for key in check_dict_solvents:
        check_dict_solvents[key] *= scaling_factor

    price_per_precatalyst_base_solvent = []
    for precatalyst, base, solvent in zip(precatalysts, bases, solvents):
        minus = (
            check_dict_precatalysts[precatalyst]
            + check_dict_bases[base]
            + check_dict_solvents[solvent]
        )
        price_per_precatalyst_base_solvent.append(minus)
        check_dict_precatalysts[precatalyst] = 0
        check_dict_bases[base] = 0
        check_dict_solvents[solvent] = 0

    return price_per_precatalyst_base_solvent


def update_price_dict_ligands(price_dict, NEW_LIGANDS):
    """
    This function is for 2_greedy_update_costs
    """
    NEW_LIGANDS = np.unique(NEW_LIGANDS)
    for ligand in NEW_LIGANDS:
        price_dict[ligand] = 0
    return price_dict


def update_price_dict_all(price_dict, NEW_LIGANDS, NEW_BASES, NEW_SOLVENTS):
    NEW_LIGANDS = np.unique(NEW_LIGANDS)
    NEW_BASES = np.unique(NEW_BASES)
    NEW_SOLVENTS = np.unique(NEW_SOLVENTS)
    for ligand in NEW_LIGANDS:
        price_dict["ligands"][ligand] = 0
    for base in NEW_BASES:
        price_dict["bases"][base] = 0
    for solvent in NEW_SOLVENTS:
        price_dict["solvents"][solvent] = 0
    return price_dict


def update_X_y(X, y, cands, y_cands_BO, inds):
    """
    Appended the new batch of molecules to the training data for the next
    iteration of the BO.
    """
    X, y = np.concatenate((X, cands)), np.concatenate((y, y_cands_BO[inds, :]))
    return X, y


def create_aligned_transposed_price_table(price_dict):
    """
    Creates a transposed table from a dictionary of compound prices, numbering the compounds in a canonical order.
    The table is formatted with aligned columns for better readability in terminal.

    Parameters:
    - price_dict: A dictionary with compound strings as keys and prices as values.

    Returns:
    A string representing the aligned transposed table of compounds and their prices.
    """
    # Check if all values are zero
    all_zero = all(value == 0 for value in price_dict.values())

    # If all values are zero, return a specific message.
    if all_zero:
        return "Bought all ligands"

    # Sort the dictionary to ensure canonical order and extract only prices for transposing
    sorted_prices = [
        price for _, price in sorted(price_dict.items(), key=lambda item: item[0])
    ]
    # Calculate column width based on the largest ligand number
    col_width = max(len(f"Ligand {len(sorted_prices)}"), len("Price"))
    # Create the header with aligned column titles
    header = " | ".join(
        [f"Ligand {idx+1}".ljust(col_width) for idx in range(len(sorted_prices))]
    )
    # Create the row with aligned prices
    row = " | ".join([f"{price}".ljust(col_width) for price in sorted_prices])
    # Combine header and row into a single string with a divider line
    divider = "-" * len(header)
    return "\n".join([header, divider, row])


def data_dict_BO_LIGAND(
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
    SURROGATE,
    AQCFCT,
    EXP_DONE_BO,
    EXP_CANDIDATE_BO,
):
    """
    Create a dictionary with all the data needed for the BO in scenario 2.
    """

    BO_data = {
        "model": model,
        "y_best_BO": y_best_BO,
        "scaler_y": scaler_y,
        "X": X,
        "y": y,
        "N_train": len(X),
        "X_candidate_BO": X_candidate_BO,
        "y_candidate_BO": y_candidate_BO,
        "LIGANDS_candidate_BO": LIGANDS_candidate_BO,
        "y_better_BO": y_better_BO,
        "price_dict_BO": price_dict_BO,
        "running_costs_BO": running_costs_BO,
        "bounds_norm": bounds_norm,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_BATCH_COST": MAX_BATCH_COST,
        "surrogate": SURROGATE,
        "acq_func": AQCFCT,
        "EXP_DONE_BO": [],
        "EXP_CANDIDATE_BO": EXP_CANDIDATE_BO,
    }

    return BO_data


def create_data_dict_BO_2B(
    model,
    y_best_BO,
    scaler_y,
    X,
    y,
    X_candidate_BO,
    y_candidate_BO,
    LIGANDS_candidate_BO,
    y_better_BO,
    price_dict_BO_ligands,
    price_dict_BO_additives,
    ADDITIVES_candidate_BO,
    running_costs_BO,
    bounds_norm,
    BATCH_SIZE,
    MAX_BATCH_COST,
    SURROGATE,
    AQCFCT,
):
    BO_data = {
        "model": model,
        "y_best_BO": y_best_BO,
        "scaler_y": scaler_y,
        "X": X,
        "y": y,
        "N_train": len(X),
        "X_candidate_BO": X_candidate_BO,
        "y_candidate_BO": y_candidate_BO,
        "LIGANDS_candidate_BO": LIGANDS_candidate_BO,
        "y_better_BO": y_better_BO,
        "price_dict_BO_ligands": price_dict_BO_ligands,
        "price_dict_BO_additives": price_dict_BO_additives,
        "ADDITIVES_candidate_BO": ADDITIVES_candidate_BO,
        "running_costs_BO": running_costs_BO,
        "bounds_norm": bounds_norm,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_BATCH_COST": MAX_BATCH_COST,
        "surrogate": SURROGATE,
        "acq_func": AQCFCT,
    }

    return BO_data


def data_dict_BO_LIGAND_BASE_SOLVENT(
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
    MAX_BATCH_COST,
    SURROGATE,
    AQCFCT,
    EXP_DONE_BO,
    EXP_CANDIDATE_BO,
):
    BO_data = {
        "model": model,
        "y_best_BO": y_best_BO,
        "scaler_y": scaler_y,
        "X": X,
        "y": y,
        "N_train": len(X),
        "X_candidate_BO": X_candidate_BO,
        "y_candidate_BO": y_candidate_BO,
        "y_better_BO": y_better_BO,
        "price_dict_BO_precatalysts": price_dict_BO_precatalysts,
        "price_dict_BO_bases": price_dict_BO_bases,
        "price_dict_BO_solvents": price_dict_BO_solvents,
        "PRECATALYSTS_candidate_BO": PRECATALYSTS_candidate_BO,
        "BASES_candidate_BO": BASES_candidate_BO,
        "SOLVENTS_candidate_BO": SOLVENTS_candidate_BO,
        "running_costs_BO": running_costs_BO,
        "bounds_norm": bounds_norm,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_BATCH_COST": MAX_BATCH_COST,
        "surrogate": SURROGATE,
        "acq_func": AQCFCT,
        "EXP_DONE_BO": [],
        "EXP_CANDIDATE_BO": EXP_CANDIDATE_BO,
    }

    return BO_data


def data_dict_RS_LIGAND_BASE_SOLVENT(
    y_candidate_RANDOM,
    y_best_RANDOM,
    PRECATALYSTS_candidate_RANDOM,
    BASES_candidate_RANDOM,
    SOLVENTS_candidate_RANDOM,
    price_dict_RANDOM_precatalysts,
    price_dict_RANDOM_bases,
    price_dict_RANDOM_solvents,
    BATCH_SIZE,
    MAX_BATCH_COST,
    y_better_RANDOM,
    running_costs_RANDOM,
    EXP_DONE_RANDOM,
    EXP_CANDIDATE_RANDOM,
):
    RANDOM_data = {
        "y_candidate_RANDOM": y_candidate_RANDOM,
        "y_best_RANDOM": y_best_RANDOM,
        "PRECATALYSTS_candidate_RANDOM": PRECATALYSTS_candidate_RANDOM,
        "BASES_candidate_RANDOM": BASES_candidate_RANDOM,
        "SOLVENTS_candidate_RANDOM": SOLVENTS_candidate_RANDOM,
        "price_dict_RANDOM_precatalysts": price_dict_RANDOM_precatalysts,
        "price_dict_RANDOM_bases": price_dict_RANDOM_bases,
        "price_dict_RANDOM_solvents": price_dict_RANDOM_solvents,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_BATCH_COST": MAX_BATCH_COST,
        "y_better_RANDOM": y_better_RANDOM,
        "running_costs_RANDOM": running_costs_RANDOM,
        "EXP_DONE_RANDOM": [],
        "EXP_CANDIDATE_RANDOM": EXP_CANDIDATE_RANDOM,
    }

    return RANDOM_data


def create_data_dict_RS_2B(
    y_candidate_RANDOM,
    y_best_RANDOM,
    LIGANDS_candidate_RANDOM,
    ADDITIVES_candidate_RANDOM,
    price_dict_RANDOM_ligands,
    price_dict_RANDOM_additives,
    BATCH_SIZE,
    MAX_BATCH_COST,
    y_better_RANDOM,
    running_costs_RANDOM,
):
    RANDOM_data = {
        "y_candidate_RANDOM": y_candidate_RANDOM,
        "y_best_RANDOM": y_best_RANDOM,
        "LIGANDS_candidate_RANDOM": LIGANDS_candidate_RANDOM,
        "ADDITIVES_candidate_RANDOM": ADDITIVES_candidate_RANDOM,
        "price_dict_RANDOM_ligands": price_dict_RANDOM_ligands,
        "price_dict_RANDOM_additives": price_dict_RANDOM_additives,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_BATCH_COST": MAX_BATCH_COST,
        "y_better_RANDOM": y_better_RANDOM,
        "running_costs_RANDOM": running_costs_RANDOM,
    }
    return RANDOM_data


def create_data_dict_RS(
    y_candidate_RANDOM,
    y_best_RANDOM,
    costs_RANDOM,
    BATCH_SIZE,
    MAX_BATCH_COST,
    y_better_RANDOM,
    running_costs_RANDOM,
):
    RANDOM_data = {
        "y_candidate_RANDOM": y_candidate_RANDOM,
        "y_best_RANDOM": y_best_RANDOM,
        "costs_RANDOM": costs_RANDOM,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_BATCH_COST": MAX_BATCH_COST,
        "y_better_RANDOM": y_better_RANDOM,
        "running_costs_RANDOM": running_costs_RANDOM,
    }

    return RANDOM_data


def data_dict_RS_LIGAND(
    y_candidate_RANDOM,
    y_best_RANDOM,
    LIGANDS_candidate_RANDOM,
    price_dict_RANDOM,
    BATCH_SIZE,
    MAX_BATCH_COST,
    y_better_RANDOM,
    running_costs_RANDOM,
    EXP_DONE_RANDOM,
    EXP_CANDIDATE_RANDOM,
):
    RANDOM_data = {
        "y_candidate_RANDOM": y_candidate_RANDOM,
        "y_best_RANDOM": y_best_RANDOM,
        "LIGANDS_candidate_RANDOM": LIGANDS_candidate_RANDOM,
        "price_dict_RANDOM": price_dict_RANDOM,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_BATCH_COST": MAX_BATCH_COST,
        "y_better_RANDOM": y_better_RANDOM,
        "running_costs_RANDOM": running_costs_RANDOM,
        "EXP_DONE_RANDOM": [],
        "EXP_CANDIDATE_RANDOM": EXP_CANDIDATE_RANDOM,
    }
    return RANDOM_data

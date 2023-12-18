# Standard library imports
import copy as cp
import itertools
import os
import pickle
import math

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import use as mpl_use

# Specific module imports
from itertools import combinations
from rdkit import Chem
from rdkit.Chem import AllChem
import pdb

mpl_use("Agg")  # Set the matplotlib backend for plotting


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


"""
Functions for plotting
"""


def plot_utility_BO_vs_RS(y_better_BO_ALL, y_better_RANDOM_ALL, name="utility.png"):
    """
    Plot the utility of the BO vs RS (Random Search) for each iteration.
    """
    # create subfolder "./figures" if it does not exist
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


"""
Function for the greedy batch selection
"""


def select_batch(suggested_costs, MAX_BATCH_COST, BATCH_SIZE):
    """
    Selects a batch of molecules from a list of suggested molecules that have the lowest indices
    while meeting the constraints of the maximum cost and batch size.
    """

    n = len(suggested_costs)
    # Check if BATCH_SIZE is larger than the length of the array, if so return None
    if BATCH_SIZE > n:
        return []
    valid_combinations = []
    # Find all valid combinations that meet the cost condition
    for indices in combinations(range(n), BATCH_SIZE):
        if sum(suggested_costs[i] for i in indices) <= MAX_BATCH_COST:
            valid_combinations.append(indices)
    # If there are no valid combinations, return None
    if not valid_combinations:
        return []
    # Select the combination with the lowest indices
    best_indices = min(
        valid_combinations, key=lambda x: tuple(suggested_costs[i] for i in x)
    )
    return list(best_indices)


def check_success(cheap_indices, indices):
    if cheap_indices == []:
        return cheap_indices, False
    else:
        cheap_indices = indices[cheap_indices]
        return cheap_indices, True


# savepkl file
def save_pkl(file, name):
    with open(name, "wb") as f:
        pickle.dump(file, f)


# load pkl file
def load_pkl(name):
    with open(name, "rb") as f:
        return pickle.load(f)


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


def find_smallest_nonzero(arr):
    # Filter out the non-zero values and find the minimum among them
    nonzero_values = [x for x in arr if x != 0]
    return min(nonzero_values) if nonzero_values else 1


def compute_price_acquisition_ligands_price_per_acqfct(NEW_LIGANDS, price_dict):
    """
    Only used for gibbon_search_modified_all_per_price!

    Computes the price for the batch. When a ligand is already in the inventory
    it will have a price of 1 to make sure the acfct value is not divided by 0.
    """

    check_dict = cp.deepcopy(price_dict)
    test_this = np.array(list(check_dict.values()))

    max_value = max(test_this)

    # divide all values by the smallest nonzero value
    for key in check_dict:
        if check_dict[key] != 0:
            check_dict[key] = check_dict[key] / max_value

    min_value = find_smallest_nonzero(test_this)
    # pdb.set_trace()
    price_per_ligand = []
    for ligand in NEW_LIGANDS:
        # test_this = np.array(list(check_dict.values()))
        # min_value = find_smallest_nonzero(test_this)
        try:
            denominator = min_value + np.log(check_dict[ligand])
        except:
            pdb.set_trace()
        price_per_ligand.append(denominator)
        # price_per_ligand.append(1+check_dict[ligand]-min_value)
        check_dict[ligand] = min_value

    price_per_ligand = np.array(price_per_ligand)

    return price_per_ligand


def function_cost(NEW_LIGANDS, price_dict):
    check_dict_ligands = cp.deepcopy(price_dict)

    price_per_ligand = []
    for ligand in NEW_LIGANDS:
        denominator = 1 + check_dict_ligands[ligand]
        price_per_ligand.append(denominator)
        check_dict_ligands[ligand] = 0

    price_per_ligand = np.array(price_per_ligand)

    return price_per_ligand


def compute_price_acquisition_ligands_price_per_acqfct_B1(
    NEW_LIGANDS, NEW_ADDITIVES, price_dict_ligands, price_dict_additives
):
    """
    Only used for gibbon_search_modified_all_per_price!

    Computes the price for the batch. When a ligand is already in the inventory
    it will have a price of 1 to make sure the acfct value is not divided by 0.
    """

    check_dict_ligands = cp.deepcopy(price_dict_ligands)
    check_dict_additives = cp.deepcopy(price_dict_additives)
    test_this_ligands = np.array(list(check_dict_ligands.values()))
    test_this_additives = np.array(list(check_dict_additives.values()))

    max_value_ligands = max(test_this_ligands)
    max_value_additives = max(test_this_additives)
    maxi_norm = min(max_value_ligands, max_value_additives)
    # divide all values by the smallest nonzero value
    for key in check_dict_ligands:
        if check_dict_ligands[key] != 0:
            check_dict_ligands[key] = check_dict_ligands[key] / maxi_norm

    for key in check_dict_additives:
        if check_dict_additives[key] != 0:
            check_dict_additives[key] = check_dict_additives[key] / maxi_norm

    min_value_ligands = find_smallest_nonzero(test_this_ligands) / maxi_norm
    min_value_additives = find_smallest_nonzero(test_this_additives) / maxi_norm

    price_per_ligand_additive = []
    for ligand, additive in zip(NEW_LIGANDS, NEW_ADDITIVES):
        try:
            denominator = (
                min_value_ligands
                + min_value_additives
                + np.log(check_dict_ligands[ligand])
                + np.log(check_dict_additives[additive])
            )
        except:
            pdb.set_trace()
        price_per_ligand_additive.append(denominator)
        check_dict_ligands[ligand] = min_value_ligands
        check_dict_additives[additive] = min_value_additives

    price_per_ligand_additive = np.array(price_per_ligand_additive)

    return price_per_ligand_additive


def function_cost_B(
    NEW_LIGANDS, NEW_ADDITIVES, price_dict_ligands, price_dict_additives
):
    """
    Only used for gibbon_search_modified_all_per_price!

    Computes the price for the batch. When a ligand is already in the inventory
    it will have a price of 1 to make sure the acfct value is not divided by 0.
    """

    check_dict_ligands = cp.deepcopy(price_dict_ligands)
    check_dict_additives = cp.deepcopy(price_dict_additives)

    price_per_ligand_additive = []
    for ligand, additive in zip(NEW_LIGANDS, NEW_ADDITIVES):
        denominator = 1 + check_dict_ligands[ligand] + check_dict_additives[additive]
        price_per_ligand_additive.append(denominator)
        check_dict_ligands[ligand] = 0
        check_dict_additives[additive] = 0

    return price_per_ligand_additive


def compute_price_acquisition_ligands_price_per_acqfct2(NEW_LIGANDS, price_dict):
    """
    Only used for gibbon_search_modified_all_per_price!

    Computes the price for the batch. When a ligand is already in the inventory
    it will have a price of 1 to make sure the acfct value is not divided by 0.
    """

    check_dict = cp.deepcopy(price_dict)
    test_this = np.array(list(check_dict.values()))

    price_per_ligand = []
    for ligand in NEW_LIGANDS:
        # pdb.set_trace()
        # check_dict[ligand]
        # test_this = np.array(list(check_dict.values()))
        # min_value = find_smallest_nonzero(test_this)
        try:
            denominator = max(test_this) + np.log(check_dict[ligand])
            # if check_dict[ligand] > 100:
            #    pdb.set_trace()
            # print(denominator)
        except:
            pdb.set_trace()
        price_per_ligand.append(denominator)
        # price_per_ligand.append(1+check_dict[ligand]-min_value)
        check_dict[ligand] = 1

    price_per_ligand = np.array(price_per_ligand)

    return price_per_ligand


def find_optimal_batch(
    price_dict_BO, NEW_LIGANDS, original_indices, BATCH_SIZE, MAX_BATCH_COST
):
    """
    Find the optimal batch of ligands that fulfills the price requirements.

    :param price_dict_BO: Dictionary of ligand prices.
    :param NEW_LIGANDS: Array of ligands.
    :param original_indices: The original indices of the ligands.
    :param BATCH_SIZE: Size of each batch.
    :param MAX_BATCH_COST: Maximum allowed cost for a batch.
    :return: Indices of the optimal batch of ligands.
    """
    # Generate 1000 unique permutations
    new_range = np.arange(len(NEW_LIGANDS))
    permutations = list(itertools.permutations(new_range, BATCH_SIZE))
    # list(itertools.permutations(NEW_LIGANDS, BATCH_SIZE))
    if len(permutations) > 1000:
        permutations = permutations[:1000]
    permutations = [list(perm) for perm in permutations]

    BATCH_LIGANDS = [NEW_LIGANDS[perm] for perm in permutations]
    BATCH_INDICES = [original_indices[perm] for perm in permutations]
    BATCH_PRICE = []
    for batch in BATCH_LIGANDS:
        curr_price = 0
        check_dict = cp.deepcopy(price_dict_BO)
        for lig in batch:
            curr_price += check_dict[lig]
            check_dict[lig] = 0

        BATCH_PRICE.append(curr_price)

    BATCH_PRICE = np.array(BATCH_PRICE)

    # find where BATCH_PRICE is smaller than MAX_BATCH_COST
    good_batches = np.where(BATCH_PRICE <= MAX_BATCH_COST)[0]
    # Todo select the batch where originally most indices where from left of original indices
    if len(good_batches) == 0:
        return [], False, 0, []
    else:
        best_batch = good_batches[0]
        best_batch_indices = BATCH_INDICES[best_batch]
        best_price = BATCH_PRICE[best_batch]
        best_LIGANDS = BATCH_LIGANDS[best_batch]
        return best_batch_indices, True, best_price, best_LIGANDS


def compute_price_acquisition_all(NEW_LIGANDS, NEW_BASES, NEW_SOLVENTS, price_dict):
    """
    This function is for 2_greedy_update_costs where all prices are updated: ligand, base and solvent
    """

    price_acquisition = 0
    check_dict = cp.deepcopy(price_dict)
    for ind, ligand in enumerate(np.unique(NEW_LIGANDS)):
        price_acquisition += check_dict["ligands"][ligand]
        check_dict["ligands"][ligand] = 0

    for ind, base in enumerate(np.unique(NEW_BASES)):
        price_acquisition += check_dict["bases"][base]
        check_dict["bases"][base] = 0

    for ind, solvent in enumerate(np.unique(NEW_SOLVENTS)):
        price_acquisition += check_dict["solvents"][solvent]
        check_dict["solvents"][solvent] = 0

    price_per_all = []
    check_dict = cp.deepcopy(price_dict)
    for ligand, base, solvent in zip(NEW_LIGANDS, NEW_BASES, NEW_SOLVENTS):
        price_per_all.append(
            price_dict["ligands"][ligand]
            + price_dict["bases"][base]
            + price_dict["solvents"][solvent]
        )
        check_dict["ligands"][ligand] = 0
        check_dict["bases"][base] = 0
        check_dict["solvents"][solvent] = 0

    price_per_all = np.array(price_per_all)

    return price_acquisition, price_per_all


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


def create_data_dict_BO_2A(
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
    AQCFCT
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
    AQCFCT
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


def create_data_dict_BO_1(
    model,
    y_best_BO,
    scaler_y,
    X,
    y,
    X_candidate_BO,
    y_candidate_BO,
    y_better_BO,
    costs_BO,
    running_costs_BO,
    bounds_norm,
    BATCH_SIZE,
    MAX_BATCH_COST,
    SURROGATE,
    AQCFCT
):
    """
    For scenario 1
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
        "y_better_BO": y_better_BO,
        "costs_BO": costs_BO,
        "running_costs_BO": running_costs_BO,
        "bounds_norm": bounds_norm,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_BATCH_COST": MAX_BATCH_COST,
        "surrogate": SURROGATE,
        "acq_func": AQCFCT,
    }
    return BO_data


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


def create_data_dict_RS_2A(
    y_candidate_RANDOM,
    y_best_RANDOM,
    LIGANDS_candidate_RANDOM,
    price_dict_RANDOM,
    BATCH_SIZE,
    MAX_BATCH_COST,
    y_better_RANDOM,
    running_costs_RANDOM,
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
    }
    return RANDOM_data


class Budget_schedule:
    def __init__(self, schedule="constant"):
        self.schedule = schedule

    def constant(self, iteration):
        return 1

    def increasing(self, iteration):
        return iteration + 1

    def decreasing(self, iteration):
        return 1 / (iteration + 1)

    def adaptive(self, iteration):
        if iteration <= 15:
            return 1
        else:
            return (iteration - 14) + 1

    def adaptive_2(self, iteration):
        if iteration <= 30:
            return 1
        elif iteration <= 60:
            return (iteration - 30) + 1

    def get_factor(self, iteration):
        if self.schedule == "constant":
            return self.constant(iteration)
        elif self.schedule == "increasing":
            return self.increasing(iteration)
        elif self.schedule == "decreasing":
            return self.decreasing(iteration)
        elif self.schedule == "adaptive":
            return self.adaptive(iteration)
        elif self.schedule == "adaptive_2":
            return self.adaptive_2(iteration)
        else:
            print("Schedule not implemented.")
            exit()

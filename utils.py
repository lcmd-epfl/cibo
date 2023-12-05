import copy as cp
import itertools
import math
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from itertools import combinations
from matplotlib import use as mpl_use
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler

mpl_use("Agg")  # Set the matplotlib backend


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


def convert2pytorch(X, y):
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float().reshape(-1, 1)
    return X, y


def check_entries(array_of_arrays):
    for array in array_of_arrays:
        for item in array:
            if item < 0 or item > 1:
                return False
    return True


class FingerprintGenerator:
    def __init__(self, nBits=512, radius=2):
        self.nBits = nBits
        self.radius = radius

    def featurize(self, smiles_list):
        fingerprints = []
        for smiles in smiles_list:
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
                print(f"Could not generate a molecule from SMILES: {smiles}")
                fingerprints.append(np.array([None]))
        return np.array(fingerprints)


class Evaluation_data:
    def __init__(self, dataset, init_size, prices, init_strategy="values"):
        self.dataset = dataset
        self.init_strategy = init_strategy
        self.init_size = init_size
        self.prices = prices

        self.ECFP_size = 512
        self.radius = 2

        self.ftzr = FingerprintGenerator(nBits=self.ECFP_size, radius=self.radius)
        self.get_raw_dataset()

        rep_size = self.X.shape[1]
        self.bounds_norm = torch.tensor([[0] * rep_size, [1] * rep_size])
        self.bounds_norm = self.bounds_norm.to(dtype=torch.float32)

        if not check_entries(self.X):
            print("###############################################")
            print(
                "Entries of X are not between 0 and 1. Adding MinMaxScaler to the pipeline."
            )
            print("###############################################")

            self.scaler_X = MinMaxScaler()
            self.X = self.scaler_X.fit_transform(self.X)

    def get_raw_dataset(self):
        if self.dataset == "freesolv":
            try:
                import deepchem as dc
            except:
                print("DeepChem not installed.")
                exit()
            _, datasets, _ = dc.molnet.load_sampl(
                featurizer=self.ftzr, splitter="random", transformers=[]
            )
            train_dataset, valid_dataset, test_dataset = datasets

            X_train = train_dataset.X
            y_train = train_dataset.y[:, 0]
            X_valid = valid_dataset.X
            y_valid = valid_dataset.y[:, 0]
            X_test = test_dataset.X
            y_test = test_dataset.y[:, 0]

            self.X = np.concatenate((X_train, X_valid, X_test))
            self.y = np.concatenate((y_train, y_valid, y_test))

            random_inds = np.random.permutation(len(self.X))
            self.X = self.X[random_inds]
            self.y = self.y[random_inds]

        # TODO: include this dataset with more reastic prices
        # https://github.com/doyle-lab-ucla/edboplus/blob/main/examples/publication/BMS_yield_cost/data/PCI_PMI_cost_full_update.csv
        # https://chemrxiv.org/engage/chemrxiv/article-details/62f6966269f3a5df46b5584b
        elif self.dataset == "BMS":
            dataset_url = "https://raw.githubusercontent.com/doyle-lab-ucla/edboplus/main/examples/publication/BMS_yield_cost/data/PCI_PMI_cost_full.csv"
            # irrelevant: Time_h , Nucleophile,Nucleophile_Equiv, Ligand_Equiv
            self.data = pd.read_csv(dataset_url)
            self.data = self.data.dropna()
            self.data = self.data.sample(frac=1).reset_index(drop=True)

            self.data["Base_SMILES"] = inchi_to_smiles(self.data["Base_inchi"].values)
            self.data["Ligand_SMILES"] = inchi_to_smiles(
                self.data["Ligand_inchi"].values
            )
            self.data["Solvent_SMILES"] = inchi_to_smiles(
                self.data["Solvent_inchi"].values
            )
            
            col_0_base = self.ftzr.featurize(self.data["Base_SMILES"])
            col_1_ligand = self.ftzr.featurize(self.data["Ligand_SMILES"])
            col_2_solvent = self.ftzr.featurize(self.data["Solvent_SMILES"])
            col_3_concentration = self.data["Concentration"].to_numpy().reshape(-1, 1)
            col_4_temperature = self.data["Temp_C"].to_numpy().reshape(-1, 1)

            self.X = np.concatenate(
                [
                    col_0_base,
                    col_1_ligand,
                    col_2_solvent,
                    col_3_concentration,
                    col_4_temperature,
                ],
                axis=1,
            )

            self.y = self.data["Yield"].to_numpy()
            self.all_ligands = self.data["Ligand_SMILES"].to_numpy()
            self.all_bases = self.data["Base_SMILES"].to_numpy()
            self.all_solvents = self.data["Solvent_SMILES"].to_numpy()
            unique_bases = np.unique(self.data["Base_SMILES"])
            unique_ligands = np.unique(self.data["Ligand_SMILES"])
            unique_solvents = np.unique(self.data["Solvent_SMILES"])
            unique_concentrations = np.unique(self.data["Concentration"])
            unique_temperatures = np.unique(self.data["Temp_C"])

            max_yield_per_ligand = np.array(
                [
                    max(self.data[self.data["Ligand_SMILES"] == unique_ligand]["Yield"])
                    for unique_ligand in unique_ligands
                ]
            )

            self.worst_ligand = unique_ligands[np.argmin(max_yield_per_ligand)]

            # make price of worst ligand 0 because already in the inventory
            self.best_ligand = unique_ligands[np.argmax(max_yield_per_ligand)]

            self.where_worst = np.array(
                self.data.index[
                    self.data["Ligand_SMILES"] == self.worst_ligand
                ].tolist()
            )

            self.feauture_labels = {
                "names": {
                    "bases": unique_bases,
                    "ligands": unique_ligands,
                    "solvents": unique_solvents,
                    "concentrations": unique_concentrations,
                    "temperatures": unique_temperatures,
                },
                "ordered_smiles": {
                    "bases": self.data["Base_SMILES"],
                    "ligands": self.data["Ligand_SMILES"],
                    "solvents": self.data["Solvent_SMILES"],
                    "concentrations": self.data["Concentration"],
                    "temperatures": self.data["Temp_C"],
                },
            }

        elif self.dataset == "ebdo_direct_arylation":
            dataset_url = "https://raw.githubusercontent.com/b-shields/edbo/master/experiments/data/direct_arylation/experiment_index.csv"
            self.data = pd.read_csv(dataset_url)
            self.data = self.data.dropna()
            self.data = self.data.sample(frac=1).reset_index(drop=True)
            col_0_base = self.ftzr.featurize(self.data["Base_SMILES"])
            col_1_ligand = self.ftzr.featurize(self.data["Ligand_SMILES"])
            col_2_solvent = self.ftzr.featurize(self.data["Solvent_SMILES"])
            col_3_concentration = self.data["Concentration"].to_numpy().reshape(-1, 1)
            col_4_temperature = self.data["Temp_C"].to_numpy().reshape(-1, 1)

            self.X = np.concatenate(
                [
                    col_0_base,
                    col_1_ligand,
                    col_2_solvent,
                    col_3_concentration,
                    col_4_temperature,
                ],
                axis=1,
            )

            self.y = self.data["yield"].to_numpy()

            self.all_ligands = self.data["Ligand_SMILES"].to_numpy()
            self.all_bases = self.data["Base_SMILES"].to_numpy()
            self.all_solvents = self.data["Solvent_SMILES"].to_numpy()
            unique_bases = np.unique(self.data["Base_SMILES"])
            unique_ligands = np.unique(self.data["Ligand_SMILES"])
            unique_solvents = np.unique(self.data["Solvent_SMILES"])
            unique_concentrations = np.unique(self.data["Concentration"])
            unique_temperatures = np.unique(self.data["Temp_C"])

            max_yield_per_ligand = np.array(
                [
                    max(self.data[self.data["Ligand_SMILES"] == unique_ligand]["yield"])
                    for unique_ligand in unique_ligands
                ]
            )
            self.worst_ligand = unique_ligands[np.argmin(max_yield_per_ligand)]
            self.best_ligand = unique_ligands[np.argmax(max_yield_per_ligand)]

            self.where_worst = np.array(
                self.data.index[
                    self.data["Ligand_SMILES"] == self.worst_ligand
                ].tolist()
            )

            self.feauture_labels = {
                "names": {
                    "bases": unique_bases,
                    "ligands": unique_ligands,
                    "solvents": unique_solvents,
                    "concentrations": unique_concentrations,
                    "temperatures": unique_temperatures,
                },
                "ordered_smiles": {
                    "bases": self.data["Base_SMILES"],
                    "ligands": self.data["Ligand_SMILES"],
                    "solvents": self.data["Solvent_SMILES"],
                    "concentrations": self.data["Concentration"],
                    "temperatures": self.data["Temp_C"],
                },
            }

        elif self.dataset == "buchwald":
            dataset_url = "https://raw.githubusercontent.com/doylelab/rxnpredict/master/data_table.csv"
            # load url directly into pandas dataframe

            data = pd.read_csv(dataset_url)
            # remove rows with nan
            data = data.dropna()
            # randomly shuffly df
            data = data.sample(frac=1).reset_index(drop=True)
            unique_bases = data["base_smiles"].unique()
            unique_ligands = data["ligand_smiles"].unique()
            unique_aryl_halides = data["aryl_halide_smiles"].unique()
            unique_additives = data["additive_smiles"].unique()

            col_0_base = self.ftzr.featurize(data["base_smiles"])
            col_1_ligand = self.ftzr.featurize(data["ligand_smiles"])
            col_2_aryl_halide = self.ftzr.featurize(data["aryl_halide_smiles"])
            col_3_additive = self.ftzr.featurize(data["additive_smiles"])

            self.feauture_labels = {
                "names": {
                    "bases": unique_bases,
                    "ligands": unique_ligands,
                    "aryl_halides": unique_aryl_halides,
                    "additives": unique_additives,
                },
                "ordered_smiles": {
                    "bases": data["base_smiles"],
                    "ligands": data["ligand_smiles"],
                    "aryl_halides": data["aryl_halide_smiles"],
                    "additives": data["additive_smiles"],
                },
            }

            self.X = np.concatenate(
                [col_0_base, col_1_ligand, col_2_aryl_halide, col_3_additive],
                axis=1,
            )
            self.y = data["yield"].to_numpy()

        else:
            print("Dataset not implemented.")
            exit()

    def get_prices(self):
        if self.prices == "random":
            self.costs = np.random.randint(2, size=(len(self.X), 1))
            best_points = np.argwhere(self.y == 100.0)

            # make best point price 1 (not for free)
            for p in best_points:
                self.costs[p] = np.array([1])

        elif self.prices == "update_ligand_when_used":
            if self.dataset == "freesolv":
                print("Not implemented.")
                exit()

            elif self.dataset == "BMS":
                ligand_price_dict = {}

                # Iterate through the dataframe rows
                for index, row in self.data.iterrows():
                    ligand_smiles = row["Ligand_SMILES"]
                    ligand_price = row["Ligand_Cost"]
                    ligand_price_dict[ligand_smiles] = ligand_price

                # Print the dictionary
                self.ligand_prices = ligand_price_dict

                all_ligand_prices = []
                for ligand in self.feauture_labels["ordered_smiles"]["ligands"]:
                    all_ligand_prices.append(self.ligand_prices[ligand])
                self.costs = np.array(all_ligand_prices).reshape(-1, 1)

                # make best point price 1

            elif self.dataset == "ebdo_direct_arylation":
                self.ligand_prices = {}
                for ind, unique_ligand in enumerate(
                    self.feauture_labels["names"]["ligands"]
                ):
                    self.ligand_prices[unique_ligand] = np.random.randint(2)

                self.ligand_prices[self.worst_ligand] = 0
                self.ligand_prices[self.best_ligand] = 1

                all_ligand_prices = []
                for ligand in self.feauture_labels["ordered_smiles"]["ligands"]:
                    all_ligand_prices.append(self.ligand_prices[ligand])
                self.costs = np.array(all_ligand_prices).reshape(-1, 1)

            elif self.dataset == "buchwald":
                self.ligand_prices = {}
                for ind, unique_ligand in enumerate(
                    self.feauture_labels["names"]["ligands"]
                ):
                    self.ligand_prices[unique_ligand] = ind + 1

                all_ligand_prices = []
                for ligand in self.feauture_labels["ordered_smiles"]["ligands"]:
                    all_ligand_prices.append(self.ligand_prices[ligand])
                self.costs = np.array(all_ligand_prices).reshape(-1, 1)

        elif self.prices == "update_all_when_bought":
            if self.dataset == "ebdo_direct_arylation":
                self.ligand_prices = {}
                for ind, unique_ligand in enumerate(
                    self.feauture_labels["names"]["ligands"]
                ):
                    self.ligand_prices[unique_ligand] = np.random.randint(2)

                self.ligand_prices[self.worst_ligand] = 0
                self.ligand_prices[self.best_ligand] = 1

                self.bases_prices = {}
                for ind, unique_base in enumerate(
                    self.feauture_labels["names"]["bases"]
                ):
                    self.bases_prices[unique_base] = np.random.randint(2)

                self.solvents_prices = {}
                for ind, unique_solvent in enumerate(
                    self.feauture_labels["names"]["solvents"]
                ):
                    self.solvents_prices[unique_solvent] = np.random.randint(2)

                all_prices = []
                for base, ligand, solvent in zip(
                    self.feauture_labels["ordered_smiles"]["bases"],
                    self.feauture_labels["ordered_smiles"]["ligands"],
                    self.feauture_labels["ordered_smiles"]["solvents"],
                ):
                    all_prices.append(
                        self.ligand_prices[ligand]
                        + self.bases_prices[base]
                        + self.solvents_prices[solvent]
                    )
                self.costs = np.array(all_prices).reshape(-1, 1)

                self.all_prices_dict = {
                    "ligands": self.ligand_prices,
                    "bases": self.bases_prices,
                    "solvents": self.solvents_prices,
                }

        else:
            print("Price model not implemented.")
            exit()

    def get_init_holdout_data(self, SEED):
        random.seed(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.get_prices()

        if self.init_strategy == "values":
            """
            Select the init_size worst values and the rest randomly.
            """

            min_val = np.mean(self.y) - 0.2 * abs(np.std(self.y))
            print("min_val", min_val)

            index_worst = np.random.choice(
                np.argwhere(self.y < min_val).flatten(),
                size=self.init_size,
                replace=False,
            )
            index_others = np.setdiff1d(np.arange(len(self.y)), index_worst)
            # randomly shuffle the data
            index_others = np.random.permutation(index_others)

            X_init, y_init, costs_init = (
                self.X[index_worst],
                self.y[index_worst],
                self.costs[index_worst],
            )
            X_holdout, y_holdout, costs_holdout = (
                self.X[index_others],
                self.y[index_others],
                self.costs[index_others],
            )

            X_init, y_init = convert2pytorch(X_init, y_init)
            X_holdout, y_holdout = convert2pytorch(X_holdout, y_holdout)

            return X_init, y_init, costs_init, X_holdout, y_holdout, costs_holdout

        elif self.init_strategy == "random":
            """
            Randomly select init_size values.
            """
            indices_init = np.random.choice(
                np.arange(len(self.y)), size=self.init_size, replace=False
            )
            indices_holdout = np.array(
                [i for i in np.arange(len(self.y)) if i not in indices_init]
            )

            np.random.shuffle(indices_init)
            np.random.shuffle(indices_holdout)

            X_init, y_init = self.X[indices_init], self.y[indices_init]
            X_holdout, y_holdout = self.X[indices_holdout], self.y[indices_holdout]
            costs_init = self.costs[indices_init]
            costs_holdout = self.costs[indices_holdout]

            X_init, y_init = convert2pytorch(X_init, y_init)
            X_holdout, y_holdout = convert2pytorch(X_holdout, y_holdout)

            return X_init, y_init, costs_init, X_holdout, y_holdout, costs_holdout

        elif self.init_strategy == "half":
            idx_max_y = np.argmax(self.y)

            # Step 2: Retrieve the corresponding vector in X
            X_max_y = self.X[idx_max_y]

            # Step 3: Compute the distance between each row in X and X_max_y
            # Using Euclidean distance as an example
            distances = np.linalg.norm(self.X - X_max_y, axis=1)
            self.init_size = int(len(distances) / 2)
            indices_init = np.argsort(distances)[::-1][: self.init_size]

            # Step 5: Get the 100 entries corresponding to these indices from X and y
            X_init = self.X[indices_init]
            y_init = self.y[indices_init]

            # Step 6: Get the remaining entries
            indices_holdout = np.setdiff1d(np.arange(self.X.shape[0]), indices_init)

            np.random.shuffle(indices_holdout)
            X_holdout = self.X[indices_holdout]
            y_holdout = self.y[indices_holdout]

            self.costs = 0
            costs_init = np.zeros_like(y_init).reshape(-1, 1)
            # create an array of len y_holdout with 50 percent 0 and 50 percent 1
            costs_holdout = np.random.randint(2, size=len(y_holdout)).reshape(-1, 1)
            costs_holdout[np.argmax(y_holdout)] = np.array([1])

            X_init, y_init = convert2pytorch(X_init, y_init)
            X_holdout, y_holdout = convert2pytorch(X_holdout, y_holdout)

            return X_init, y_init, costs_init, X_holdout, y_holdout, costs_holdout

        elif self.init_strategy == "furthest":
            """
            Select molecules furthest away the representation of the global optimum.
            """
            idx_max_y = np.argmax(self.y)

            # Step 2: Retrieve the corresponding vector in X
            X_max_y = self.X[idx_max_y]

            # Step 3: Compute the distance between each row in X and X_max_y
            # Using Euclidean distance as an example
            distances = np.linalg.norm(self.X - X_max_y, axis=1)

            # Step 4: Sort these distances and get the indices of the 100 largest distances
            indices_init = np.argsort(distances)[::-1][: self.init_size]

            # Step 5: Get the 100 entries corresponding to these indices from X and y
            X_init = self.X[indices_init]
            y_init = self.y[indices_init]

            # Step 6: Get the remaining entries
            indices_holdout = np.setdiff1d(np.arange(self.X.shape[0]), indices_init)
            np.random.shuffle(indices_holdout)
            X_holdout = self.X[indices_holdout]
            y_holdout = self.y[indices_holdout]
            costs_init = self.costs[indices_init]
            costs_holdout = self.costs[indices_holdout]

            X_init, y_init = convert2pytorch(X_init, y_init)
            X_holdout, y_holdout = convert2pytorch(X_holdout, y_holdout)

            return X_init, y_init, costs_init, X_holdout, y_holdout, costs_holdout

        elif self.init_strategy == "worst_ligand":
            indices_init = self.where_worst[: self.init_size]
            indices_holdout = np.setdiff1d(np.arange(len(self.y)), indices_init)
            np.random.shuffle(indices_init)
            np.random.shuffle(indices_holdout)

            X_init, y_init = self.X[indices_init], self.y[indices_init]
            X_holdout, y_holdout = self.X[indices_holdout], self.y[indices_holdout]

            price_dict_init = self.ligand_prices
            price_dict_init[self.worst_ligand] = 0
            LIGANDS_INIT = self.all_ligands[indices_init]
            LIGANDS_HOLDOUT = self.all_ligands[indices_holdout]

            costs_init = self.costs[indices_init]
            costs_holdout = self.costs[indices_holdout]

            X_init, y_init = convert2pytorch(X_init, y_init)
            X_holdout, y_holdout = convert2pytorch(X_holdout, y_holdout)

            return (
                X_init,
                y_init,
                costs_init,
                X_holdout,
                y_holdout,
                costs_holdout,
                LIGANDS_INIT,
                LIGANDS_HOLDOUT,
                price_dict_init,
            )

        elif self.init_strategy == "worst_ligand_base_solvent":
            assert (
                self.dataset == "ebdo_direct_arylation"
            ), "This init strategy is only implemented for the ebdo_direct_arylation dataset."

            unique_bases = self.feauture_labels["names"]["bases"]
            unique_solvents = self.feauture_labels["names"]["solvents"]
            # start with 2 solvents, 2 bases
            # select two random bases and two random solvents
            self.bases_init = np.random.choice(unique_bases, size=2, replace=False)
            self.solvents_init = np.random.choice(
                unique_solvents, size=2, replace=False
            )

            # update the price dict

            for base in self.bases_init:
                self.all_prices_dict["bases"][base] = 0
            for solvent in self.solvents_init:
                self.all_prices_dict["solvents"][solvent] = 0
            self.all_prices_dict["ligands"][self.worst_ligand] = 0

            # select the indices of the worst ligand with the two bases and two solvents
            indices_init = np.where(
                (self.all_ligands == self.worst_ligand)
                & np.isin(self.all_bases, self.bases_init)
                & np.isin(self.all_solvents, self.solvents_init)
            )[0]
            indices_holdout = np.setdiff1d(np.arange(len(self.y)), indices_init)
            np.random.shuffle(indices_init)
            np.random.shuffle(indices_holdout)

            X_init, y_init = self.X[indices_init], self.y[indices_init]
            X_holdout, y_holdout = self.X[indices_holdout], self.y[indices_holdout]
            X_init, y_init = convert2pytorch(X_init, y_init)
            X_holdout, y_holdout = convert2pytorch(X_holdout, y_holdout)
            price_dict_init = self.all_prices_dict

            LIGANDS_INIT = self.all_ligands[indices_init]
            LIGANDS_HOLDOUT = self.all_ligands[indices_holdout]

            BASES_INIT = self.all_bases[indices_init]
            BASES_HOLDOUT = self.all_bases[indices_holdout]

            SOLVENTS_INIT = self.all_solvents[indices_init]
            SOLVENTS_HOLDOUT = self.all_solvents[indices_holdout]

            return (
                X_init,
                y_init,
                X_holdout,
                y_holdout,
                LIGANDS_INIT,
                LIGANDS_HOLDOUT,
                BASES_INIT,
                BASES_HOLDOUT,
                SOLVENTS_INIT,
                SOLVENTS_HOLDOUT,
                price_dict_init,
            )

        else:
            print("Init strategy not implemented.")
            exit()


def plot_utility_BO_vs_RS(
    y_better_BO_ALL, y_better_RANDOM_ALL, name="./figures/utility.png"
):
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
    plt.savefig(name)

    plt.clf()


def plot_costs_BO_vs_RS(
    running_costs_BO_ALL, running_costs_RANDOM_ALL, name="./figures/costs.png"
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
    plt.savefig(name)

    plt.clf()


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return None


def canonicalize_smiles_list(smiles_list):
    return [canonicalize_smiles(smiles) for smiles in smiles_list]


def check_better(y, y_best_BO):
    """
    Check if one of the molecuels in the new batch
    is better than the current best one.
    """

    if max(y)[0] > y_best_BO:
        return max(y)[0]
    else:
        return y_best_BO


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


def find_min_max_distance_and_ratio_scipy(x, vectors):
    """
    #FUNCTION concerns subfolder 3_similarity_based_costs
    (helper function for get_batch_price function)
    Calculate the minimum and maximum distance between a vector x and a set of vectors vectors.
    Parameters:
        x (numpy.ndarray): The vector x.
        vectors (numpy.ndarray): The set of vectors.
    Returns:
        tuple: The ratio between the minimum and maximum distance, the minimum distance, and the maximum distance.

    Equation for computation of the ratio:
    \[
    p(x, \text{vectors}) = \frac{\min_{i} d(x, \text{vectors}[i])}{\max \left( \max_{i,k} d(\text{vectors}[i], \text{vectors}[k]), \max_{i} d(x, \text{vectors}[i]) \right)}
    \]

    \[
    d(a, b) = \sqrt{\sum_{j=1}^{n} (a[j] - b[j])^2}
    \]
    """
    # Calculate the minimum distance between x and vectors using cdist
    dist_1 = distance.cdist([x], vectors, "euclidean")
    min_distance = np.min(dist_1)
    # Calculate the maximum distance among all vectors and x using cdist
    pairwise_distances = distance.cdist(vectors, vectors, "euclidean")
    max_distance_vectors = np.max(pairwise_distances)
    max_distance_x = np.max(dist_1)
    max_distance = max(max_distance_vectors, max_distance_x)
    # Calculate the ratio p = min_distance / max_distance
    p = min_distance / max_distance
    return p


def get_batch_price(X_train, costy_mols):
    """
    #FUNCTION concerns subfolder 3_similarity_based_costs
    Computes the total price of a batch of molecules.
    to update the price dynamically as the batch is being constructed
    for BO with synthesis at each iteration

    Parameters:
        X_train (numpy.ndarray): The training data.
        costy_mols (numpy.ndarray): The batch of molecules.
    Returns:
        float: The total price of the batch.

    e.g. if a molecule was included in the training set its price will be 0
    if a similar molecule was not included in the training set its price will be 1
    for cases in between the price will be between 0 and 1
    this is done for all costly molecules in the batch and the total price is returned
    """

    X_train_cp = cp.deepcopy(X_train)
    batch_price = 0

    for mol in costy_mols:
        costs = find_min_max_distance_and_ratio_scipy(mol, X_train_cp)
        batch_price += costs  # Update the batch price
        X_train_cp = np.vstack((X_train_cp, mol))

    return batch_price


def update_X_y(X, y, cands, y_cands_BO, inds):
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
    }

    return BO_data


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    benchmark = {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": False,
        "n_runs": 5,
        "n_iter": 15,
        "batch_size": 5,
        "max_batch_cost": 0,
        "ntrain": 200,
    }

    DATASET = Evaluation_data(
        benchmark["dataset"],
        benchmark["ntrain"],
        "update_ligand_when_used",
        init_strategy=benchmark["init_strategy"],
    )

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
    ) = DATASET.get_init_holdout_data(77)


def round_up_to_next_ten(n):
    return math.ceil(n / 10) * 10



class Budget_schedule:
    def __init__(self, schedule="constant"):
        self.schedule = schedule
    def constant(self, iteration):
        return 1
    def increasing(self, iteration):
        return iteration + 1
    def decreasing(self, iteration):
        return 1 / (iteration + 1)

    def get_factor(self, iteration):
        if self.schedule == "constant":
            return self.constant(iteration)
        elif self.schedule == "increasing":
            return self.increasing(iteration)
        elif self.schedule == "decreasing":
            return self.decreasing(iteration)
        else:
            print("Schedule not implemented.")
            exit()


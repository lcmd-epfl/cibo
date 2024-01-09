import numpy as np
import pandas as pd
import torch
import random
from utils import FingerprintGenerator, inchi_to_smiles, convert2pytorch, check_entries
from sklearn.preprocessing import MinMaxScaler
from data.buchwald import buchwald_prices
from data.baumgartner import baumgartner2019_prices
import pdb


def index_of_second_smallest(arr):
    # Convert to numpy array if not already
    arr = np.array(arr)

    # Check if array has at least two elements
    if len(arr) < 2:
        return None  # Or raise an error

    # Find the index of the minimum value
    min_index = np.argmin(arr)

    # Temporarily set the minimum value to a very high value
    original_min = arr[min_index]
    arr[min_index] = np.inf

    # Find the new minimum, which is the second smallest
    second_min_index = np.argmin(arr)

    # Revert the change to the original array
    arr[min_index] = original_min

    return second_min_index


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
        # TODO: include this dataset with more reastic prices
        # https://github.com/doyle-lab-ucla/edboplus/blob/main/examples/publication/BMS_yield_cost/data/PCI_PMI_cost_full_update.csv
        # https://chemrxiv.org/engage/chemrxiv/article-details/62f6966269f3a5df46b5584b

        if self.dataset == "BMS":
            # direct arylation reaction
            dataset_url = "https://raw.githubusercontent.com/doyle-lab-ucla/edboplus/main/examples/publication/BMS_yield_cost/data/PCI_PMI_cost_full.csv"
            # irrelevant: Time_h , Nucleophile,Nucleophile_Equiv, Ligand_Equiv
            self.data = pd.read_csv(dataset_url)
            self.data = self.data.sample(frac=1).reset_index(drop=True)
            # create a copy of the data
            data_copy = self.data.copy()
            # remove the Yield column from the copy
            data_copy.drop("Yield", axis=1, inplace=True)
            # check for duplicates
            duplicates = data_copy.duplicated().any()
            if duplicates:
                print("There are duplicates in the dataset.")
                exit()

            self.data["Ligand_Cost_fixed"] = np.ceil(
                self.data["Ligand_price.mol"].values / self.data["Ligand_MW"].values
            )

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

            self.where_worst_ligand = np.array(
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
            self.data = pd.read_csv(dataset_url)
            self.data = self.data.sample(frac=1).reset_index(drop=True)
            # randomly shuffly df
            data_copy = self.data.copy()
            # remove the Yield column from the copy
            data_copy.drop("yield", axis=1, inplace=True)
            # check for duplicates
            duplicates = data_copy.duplicated().any()

            if duplicates:
                print("There are duplicates in the dataset.")
                exit()

            col_0_base = self.ftzr.featurize(self.data["base_smiles"])
            col_1_ligand = self.ftzr.featurize(self.data["ligand_smiles"])
            col_2_aryl_halide = self.ftzr.featurize(self.data["aryl_halide_smiles"])
            col_3_additive = self.ftzr.featurize(self.data["additive_smiles"])

            self.X = np.concatenate(
                [col_0_base, col_1_ligand, col_2_aryl_halide, col_3_additive],
                axis=1,
            )

            self.y = self.data["yield"].to_numpy()
            self.all_ligands = self.data["ligand_smiles"].to_numpy()
            self.all_bases = self.data["base_smiles"].to_numpy()
            self.all_aryl_halides = (
                self.data["aryl_halide_smiles"].fillna("zero").to_numpy()
            )
            self.all_additives = self.data["additive_smiles"].fillna("zero").to_numpy()
            self.unique_bases = np.unique(self.data["base_smiles"])
            self.unique_ligands = np.unique(self.data["ligand_smiles"])
            self.unique_aryl_halides = np.unique(
                self.data["aryl_halide_smiles"].fillna("zero")
            )
            self.unique_additives = np.unique(
                self.data["additive_smiles"].fillna("zero")
            )

            max_yield_per_ligand = np.array(
                [
                    max(self.data[self.data["ligand_smiles"] == unique_ligand]["yield"])
                    for unique_ligand in self.unique_ligands
                ]
            )

            self.worst_ligand = self.unique_ligands[np.argmin(max_yield_per_ligand)]
            self.worst_base = self.unique_bases[np.argmin(max_yield_per_ligand)]
            self.worst_aryl_halide = self.unique_aryl_halides[
                np.argmin(max_yield_per_ligand)
            ]
            self.worst_additive = self.unique_additives[np.argmin(max_yield_per_ligand)]

            # make price of worst ligand 0 because already in the inventory
            self.best_ligand = self.unique_ligands[np.argmax(max_yield_per_ligand)]

            self.where_worst_ligand = np.array(
                self.data.index[
                    self.data["ligand_smiles"] == self.worst_ligand
                ].tolist()
            )

            self.where_worst_base = np.array(
                self.data.index[self.data["base_smiles"] == self.worst_base].tolist()
            )

            self.where_worst_aryl_halide = np.array(
                self.data.index[
                    self.data["aryl_halide_smiles"] == self.worst_aryl_halide
                ].tolist()
            )

            self.where_worst_additive = np.array(
                self.data.index[
                    self.data["additive_smiles"] == self.worst_additive
                ].tolist()
            )

            self.feauture_labels = {
                "names": {
                    "bases": self.unique_bases,
                    "ligands": self.unique_ligands,
                    "aryl_halides": self.unique_aryl_halides,
                    "additives": self.unique_additives,
                },
                "ordered_smiles": {
                    "bases": self.data["base_smiles"],
                    "ligands": self.data["ligand_smiles"],
                    "aryl_halides": self.data["aryl_halide_smiles"].fillna("zero"),
                    "additives": self.data["additive_smiles"].fillna("zero"),
                },
            }

            (
                self.price_dict_additives,
                self.price_dict_aryl_halides,
                self.price_dict_bases,
                self.price_dict_ligands,
            ) = buchwald_prices()

            self.cheapest_additive = np.array(list(self.price_dict_additives.keys()))[
                index_of_second_smallest(
                    np.array(list(self.price_dict_additives.values()))
                )
            ]
            self.cheapest_ligand = np.array(list(self.price_dict_ligands.keys()))[
                np.argmin(np.array(list(self.price_dict_ligands.values())))
            ]

            self.where_cheapest_additive = np.array(
                self.data.index[
                    self.data["additive_smiles"] == self.cheapest_additive
                ].tolist()
            )

            self.where_cheapest_ligand = np.array(
                self.data.index[
                    self.data["ligand_smiles"] == self.cheapest_ligand
                ].tolist()
            )

        elif self.dataset == "baumgartner":
            data, self.all_price_dicts = baumgartner2019_prices()
            self.data = data[
                "Phenethylamine"
            ]  # Benzamide, Phenethylamine, Aniline, Morpholine

            self.data = self.data.sample(frac=1).reset_index(drop=True)
            data_copy = self.data.copy()
            data_copy.drop("yield", axis=1, inplace=True)
            # check for duplicates
            duplicates = data_copy.duplicated().any()

            if duplicates:
                print("There are duplicates in the dataset.")
                exit()

            # Compounds used
            col_0_precatalyst = self.ftzr.featurize(self.data["precatalyst_smiles"])
            col_1_solvent = self.ftzr.featurize(self.data["solvent_smiles"])
            col_2_base = self.ftzr.featurize(self.data["base_smiles"])

            # More Reaction conditions
            col_3_nuc_inj = self.data["N-H nucleophile Inlet Injection (uL)"].values
            col_4_nuc_conc = self.data["N-H nucleophile concentration (M)"].values
            col_5_aryl_tri_conc = self.data["Aryl triflate concentration (M)"].values
            col_6_precat_loading = self.data["Precatalyst loading in mol%"].values
            col_7_int_std_conc = self.data[
                "Internal Standard Concentration 1-fluoronaphthalene (g/L)"
            ].values
            col_8_base_conc = self.data["Base concentration (M)"].values
            col_9_quench = self.data["Quench Outlet Injection (uL)"].values
            col_10_temp = self.data["Temperature (degC)"].values

            self.X = np.concatenate(
                [
                    col_0_precatalyst,
                    col_1_solvent,
                    col_2_base,
                    col_3_nuc_inj.reshape(-1, 1),
                    col_4_nuc_conc.reshape(-1, 1),
                    col_5_aryl_tri_conc.reshape(-1, 1),
                    col_6_precat_loading.reshape(-1, 1),
                    col_7_int_std_conc.reshape(-1, 1),
                    col_8_base_conc.reshape(-1, 1),
                    col_9_quench.reshape(-1, 1),
                    col_10_temp.reshape(-1, 1),
                ],
                axis=1,
            )

            self.y = self.data["yield"].to_numpy()
            self.all_precatalysts = self.data["precatalyst_smiles"].to_numpy()
            self.all_solvents = self.data["solvent_smiles"].to_numpy()
            self.all_bases = self.data["base_smiles"].to_numpy()

            unique_precatalysts = np.unique(self.data["precatalyst_smiles"])
            unique_solvents = np.unique(self.data["solvent_smiles"])
            unique_bases = np.unique(self.data["base_smiles"])

            max_yield_per_precatalyst = np.array(
                [
                    max(
                        self.data[
                            self.data["precatalyst_smiles"] == unique_precatalyst
                        ]["yield"]
                    )
                    for unique_precatalyst in unique_precatalysts
                ]
            )
            # pdb.set_trace()
            self.worst_precatalyst = unique_precatalysts[
                np.argmin(max_yield_per_precatalyst)
            ]

            self.where_worst_precatalyst = np.array(
                self.data.index[
                    self.data["precatalyst_smiles"] == self.worst_precatalyst
                ].tolist()
            )

            max_yield_per_bases = np.array(
                [
                    max(self.data[self.data["base_smiles"] == unique_base]["yield"])
                    for unique_base in unique_bases
                ]
            )

            self.worst_bases = unique_bases[np.argmin(max_yield_per_bases)]
            self.where_worst_bases = np.array(
                self.data.index[self.data["base_smiles"] == self.worst_bases].tolist()
            )

            self.feauture_labels = {
                "names": {
                    "precatalysts": unique_precatalysts,
                    "solvents": unique_solvents,
                    "bases": unique_bases,
                },
                "ordered_smiles": {
                    "precatalysts": self.data["precatalyst_smiles"],
                    "solvents": self.data["solvent_smiles"],
                    "bases": self.data["base_smiles"],
                },
            }

            self.price_dict_precatalyst = self.all_price_dicts["precatalyst"]
            self.price_dict_solvent = self.all_price_dicts["solvent"]
            self.price_dict_base = self.all_price_dicts["base"]

        elif self.dataset == "TwoDimFct":
            self.data = TwoDimFct()
            self.data = self.data.data
            self.data = self.data.sample(frac=1).reset_index(drop=True)

            self.X = self.data[["x", "y"]].to_numpy()
            self.y = self.data["function_value"].to_numpy()
            self.costs = self.data["cost"].to_numpy().reshape(-1, 1)

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
            if self.dataset == "BMS":
                ligand_price_dict = {}

                # Iterate through the dataframe rows
                for index, row in self.data.iterrows():
                    ligand_smiles = row["Ligand_SMILES"]
                    ligand_price = row["Ligand_Cost_fixed"]
                    ligand_price_dict[ligand_smiles] = ligand_price

                # Print the dictionary
                self.ligand_prices = ligand_price_dict

                all_ligand_prices = []
                for ligand in self.feauture_labels["ordered_smiles"]["ligands"]:
                    all_ligand_prices.append(self.ligand_prices[ligand])
                self.costs = np.array(all_ligand_prices).reshape(-1, 1)

                # make best point price 1

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

            elif self.dataset == "TwoDimFct":
                self.ligand_prices = {}
                ind = 0

                for unique_ligand, price in zip(self.X, self.costs):
                    self.ligand_prices[ind] = price[0]
                    ind += 1

        elif self.prices == "update_all_when_bought":
            if self.dataset == "buchwald":
                return (
                    self.price_dict_additives,
                    self.price_dict_aryl_halides,
                    self.price_dict_bases,
                    self.price_dict_ligands,
                )

            elif self.dataset == "baumgartner":
                return (
                    self.price_dict_precatalyst,
                    self.price_dict_solvent,
                    self.price_dict_base,
                )
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

            if self.dataset != "TwoDimFct":
                return X_init, y_init, costs_init, X_holdout, y_holdout, costs_holdout
            else:
                for ind in index_worst:
                    self.data.loc[ind, "cost"] = 0
                    self.ligand_prices[ind] = 0
                return (
                    X_init,
                    y_init,
                    X_holdout,
                    y_holdout,
                    index_worst,
                    index_others,
                    self.ligand_prices,
                )
        elif self.init_strategy == "TwoDim":
            target_point = np.array([0, 0.5])

            # Calculate the Euclidean distances
            distances = np.sqrt(np.sum((self.X - target_point) ** 2, axis=1))

            # Find the index of the minimum distance
            closest_index = np.argmin(distances)

            X_init, y_init, costs_init = (
                self.X[closest_index],
                self.y[closest_index],
                self.costs[closest_index],
            )

            index_others = np.setdiff1d(np.arange(len(self.y)), closest_index)
            # randomly shuffle the data
            index_others = np.random.permutation(index_others)

            X_holdout, y_holdout, costs_holdout = (
                self.X[index_others],
                self.y[index_others],
                self.costs[index_others],
            )

            # reshape to 2d array
            X_init = X_init.reshape(1, -1)
            X_holdout = X_holdout.reshape(-1, 2)
            y_init = y_init.reshape(1, -1)
            y_holdout = y_holdout.reshape(-1, 1)
            X_init, y_init = convert2pytorch(X_init, y_init)
            X_holdout, y_holdout = convert2pytorch(X_holdout, y_holdout)

            self.ligand_prices[closest_index] = 0

            return (
                X_init,
                y_init,
                X_holdout,
                y_holdout,
                closest_index,
                index_others,
                self.ligand_prices,
            )

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
            indices_init = self.where_worst_ligand[: self.init_size]
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

        elif self.init_strategy == "worst_ligand_and_more":
            if self.dataset == "BMS":
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

            elif self.dataset == "buchwald":
                # TODO: implement this for buchwald
                # indices_init = np.array(
                #    list(set(self.where_worst_ligand) & set(self.where_worst_additive))
                # )
                indices_init = np.array(
                    list(set(self.where_worst_ligand) & set(self.where_worst_additive))
                )

                indices_holdout = np.setdiff1d(np.arange(len(self.y)), indices_init)
                np.random.shuffle(indices_init)
                np.random.shuffle(indices_holdout)

                X_init, y_init = self.X[indices_init], self.y[indices_init]
                X_holdout, y_holdout = self.X[indices_holdout], self.y[indices_holdout]

                self.price_dict_ligands[self.worst_ligand] = 0
                self.price_dict_additives[self.worst_additive] = 0

                LIGANDS_INIT = self.all_ligands[indices_init]
                LIGANDS_HOLDOUT = self.all_ligands[indices_holdout]

                ADDITIVES_INIT = self.all_additives[indices_init]
                ADDITIVES_HOLDOUT = self.all_additives[indices_holdout]

                X_init, y_init = convert2pytorch(X_init, y_init)
                X_holdout, y_holdout = convert2pytorch(X_holdout, y_holdout)

                return (
                    X_init,
                    y_init,
                    X_holdout,
                    y_holdout,
                    LIGANDS_INIT,
                    LIGANDS_HOLDOUT,
                    ADDITIVES_INIT,
                    ADDITIVES_HOLDOUT,
                    self.price_dict_ligands,
                    self.price_dict_additives,
                )

            elif self.dataset == "baumgartner":
                indices_init = np.array(
                    list(
                        set(self.where_worst_precatalyst) & set(self.where_worst_bases)
                    )
                )

                # self.where_worst_precatalyst[: self.init_size]
                # self.lf.where_worst_bases
                indices_holdout = np.setdiff1d(np.arange(len(self.y)), indices_init)

                np.random.shuffle(indices_holdout)

                X_init, y_init = self.X[indices_init], self.y[indices_init]
                X_holdout, y_holdout = self.X[indices_holdout], self.y[indices_holdout]

                self.price_dict_precatalyst[self.worst_precatalyst] = 0
                self.price_dict_base[self.worst_bases] = 0

                # solvent included by that
                self.worst_solvents = np.unique(
                    self.feauture_labels["ordered_smiles"]["solvents"][
                        indices_init
                    ].values
                )

                for solv in self.worst_solvents:
                    self.price_dict_solvent[solv] = 0

                PRECATALYSTS_INIT = self.all_precatalysts[indices_init]
                PRECATALYSTS_HOLDOUT = self.all_precatalysts[indices_holdout]

                BASES_INIT = self.all_bases[indices_init]
                BASES_HOLDOUT = self.all_bases[indices_holdout]

                SOLVENTS_INIT = self.all_solvents[indices_init]
                SOLVENTS_HOLDOUT = self.all_solvents[indices_holdout]

                X_init, y_init = convert2pytorch(X_init, y_init)
                X_holdout, y_holdout = convert2pytorch(X_holdout, y_holdout)

                return (
                    X_init,
                    y_init,
                    X_holdout,
                    y_holdout,
                    PRECATALYSTS_INIT,
                    PRECATALYSTS_HOLDOUT,
                    BASES_INIT,
                    BASES_HOLDOUT,
                    SOLVENTS_INIT,
                    SOLVENTS_HOLDOUT,
                    self.price_dict_precatalyst,
                    self.price_dict_base,
                    self.price_dict_solvent,
                )

        else:
            print("Init strategy not implemented.")
            exit()


class TwoDimFct:
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples

        # Define range for input
        r_min, r_max = -5.0, 5.0
        # Sample input range uniformly at 0.1 increments
        random_x = np.random.uniform(r_min, r_max, 1000)
        random_y = np.random.uniform(r_min, r_max, 1000)
        # Compute targets
        random_results = self.objective(random_x, random_y)
        # add noise to the results
        random_results += np.random.normal(0, 0.2, len(random_results))

        random_cost = self.smooth_cost_function(random_x, random_y)
        self.data = pd.DataFrame(
            {
                "x": random_x,
                "y": random_y,
                "function_value": random_results,
                "cost": random_cost,
            }
        )

    def objective(self, x, y):
        return -(x**2 + y**2)

    def smooth_cost_function(self, x, y):
        radius = np.sqrt(x**2 + y**2)
        peak_radius = 2
        ring_width = 0.5

        angle = np.arctan2(y, x)
        gap_angle = np.pi / 6

        in_gap = (angle >= -gap_angle) & (angle <= gap_angle)  # Element-wise comparison
        cost = np.exp(-((radius - peak_radius) ** 2) / (2 * ring_width**2))
        cost[in_gap] = 0  # Set cost to 0 within the gap

        return cost


if __name__ == "__main__":
    # DATASET = Evaluation_data(
    #    "buchwald", 200, "update_all_when_bought", init_strategy="worst_ligand_and_more"
    # )
    # DATASET.get_init_holdout_data(111)

    DATASET = Evaluation_data(
        "baumgartner",
        200,
        "update_all_when_bought",
        init_strategy="worst_ligand_and_more",
    )

    DATASET.get_init_holdout_data(111)

import numpy as np
import pandas as pd
import torch
import random
from utils import FingerprintGenerator, inchi_to_smiles, convert2pytorch, check_entries
from sklearn.preprocessing import MinMaxScaler
import pdb
from data.buchwald import buchwald_prices


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

            # pdb.set_trace()

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

            elif self.dataset == "TwoDimFct":
                self.ligand_prices = {}
                ind = 0

                for unique_ligand, price in zip(self.X, self.costs):
                    self.ligand_prices[ind] = price[0]
                    ind += 1
                    # pdb.set_trace()

        elif self.prices == "update_all_when_bought":
            if self.dataset == "buchwald":
                return (
                    self.price_dict_additives,
                    self.price_dict_aryl_halides,
                    self.price_dict_bases,
                    self.price_dict_ligands,
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
                    list(
                        set(self.where_worst_ligand)
                        & set(self.where_worst_additive)
                    )
                )
                #pdb.set_trace()
                #indices_init = indices_init[:30]

                # indices_init = indices_init[: 5]
                # pdb.set_trace()
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
        additional_peaks = [
            (2, 2, 1.0),
            (-1, -1, 0.5),
            (3, -3, 0.8),
            (-2, 3, 0.7),
            (4, 0, 0.7),
            (0, -4, 0.7),
            (-2, -4, 0.7),
            (4, -2, 0.7),
            (4.5, 2, 0.7),
            (2, 4, 3),
        ]
        random_cost = self.smooth_cost_function(
            random_x, random_y, peaks=additional_peaks
        )
        # pdb.set_trace()
        self.data = pd.DataFrame(
            {
                "x": random_x,
                "y": random_y,
                "function_value": random_results,
                "cost": random_cost,
            }
        )

    def objective(self, x, y):
        return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

    def smooth_cost_function(
        self,
        x,
        y,
        square_size=3.0,
        ellipse_scale=2.0,
        radial_scale=2.5,
        peaks=None,
        cost_inside_square=1.0,
        smoothness=1,
        sinusoidal_frequency=0.5,
    ):
        """
        Even more modified cost function that adds multiple peaks to the existing complex landscape.
        """
        if peaks is None:
            peaks = [(3, 2, 2, 1.0, 0.5, 0.25)]  # Default peak if none provided

        half_size = square_size / 2
        smooth_transition = lambda d: 1 / (1 + np.exp(-smoothness * (d - half_size)))

        # Square cost component
        distance_x = np.abs(x) - half_size
        distance_y = np.abs(y) - half_size
        smooth_cost_x = smooth_transition(distance_x)
        smooth_cost_y = smooth_transition(distance_y)
        square_cost = np.minimum(smooth_cost_x, smooth_cost_y)

        # Elliptical cost component
        ellipse_cost = np.exp(-((x / ellipse_scale) ** 2 + (y / ellipse_scale) ** 2))

        # Radial cost component
        radial_distance = np.sqrt(x**2 + y**2)
        radial_cost = np.exp(-((radial_distance / radial_scale) ** 2))

        # Sinusoidal variation based on distance from the origin
        sinusoidal_variation = np.sin(sinusoidal_frequency * radial_distance)

        # Peaks
        peak_cost = 0
        for peak_x, peak_y, peak_scale in peaks:
            peak_distance = np.sqrt((x - peak_x) ** 2 + (y - peak_y) ** 2)
            peak_cost += np.exp(-((peak_distance / peak_scale) ** 2))

        # Combine all components
        combined_cost = (
            cost_inside_square
            * np.maximum(square_cost, ellipse_cost)
            * radial_cost
            * (1 + sinusoidal_variation)
            + peak_cost
        )

        return 1000 * combined_cost


if __name__ == "__main__":
    DATASET = Evaluation_data(
        "buchwald", 200, "update_all_when_bought", init_strategy="worst_ligand_and_more"
    )

    DATASET.get_init_holdout_data(111)

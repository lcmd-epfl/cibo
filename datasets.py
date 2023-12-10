import numpy as np
import pandas as pd
import torch
import random
from utils import FingerprintGenerator, inchi_to_smiles, convert2pytorch, check_entries
from sklearn.preprocessing import MinMaxScaler
import pdb

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
            #direct arylation reaction
            dataset_url = "https://raw.githubusercontent.com/doyle-lab-ucla/edboplus/main/examples/publication/BMS_yield_cost/data/PCI_PMI_cost_full.csv"
            # irrelevant: Time_h , Nucleophile,Nucleophile_Equiv, Ligand_Equiv
            self.data = pd.read_csv(dataset_url)
            self.data = self.data.sample(frac=1).reset_index(drop=True)
            #create a copy of the data
            data_copy = self.data.copy()
            #remove the Yield column from the copy
            data_copy.drop('Yield', axis=1, inplace=True)
            #check for duplicates
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
            self.data = pd.read_csv(dataset_url)
            self.data = self.data.sample(frac=1).reset_index(drop=True)
            # randomly shuffly df
            data_copy = self.data.copy()
            #remove the Yield column from the copy
            data_copy.drop('yield', axis=1, inplace=True)
            #check for duplicates
            duplicates = data_copy.duplicated().any()
            
            if duplicates:
                print("There are duplicates in the dataset.")
                exit()


            col_0_base = self.ftzr.featurize(self.data["base_smiles"])
            col_1_ligand = self.ftzr.featurize(self.data["ligand_smiles"])
            col_2_aryl_halide = self.ftzr.featurize(self.data["aryl_halide_smiles"])
            col_3_additive = self.ftzr.featurize(self.data["additive_smiles"])
            
            self.X = np.concatenate(
                [col_0_base, 
                 col_1_ligand, 
                 col_2_aryl_halide, 
                 col_3_additive
                ],
                axis=1,
            )

            self.y = self.data["yield"].to_numpy()
            self.all_ligands = self.data["ligand_smiles"].to_numpy()
            self.all_bases = self.data["base_smiles"].to_numpy()
            self.all_aryl_halides = self.data["aryl_halide_smiles"].to_numpy()
            self.all_additives = self.data["additive_smiles"].to_numpy()
            
            unique_bases = np.unique(self.data["base_smiles"])
            unique_ligands = np.unique(self.data["ligand_smiles"])
            unique_aryl_halides = np.unique(self.data["aryl_halide_smiles"].fillna(""))
            unique_additives = np.unique(self.data["additive_smiles"].fillna(""))

            max_yield_per_ligand = np.array(
                [
                    max(self.data[self.data["ligand_smiles"] == unique_ligand]["yield"])
                    for unique_ligand in unique_ligands
                ]
            )

            self.worst_ligand = unique_ligands[np.argmin(max_yield_per_ligand)]

            # make price of worst ligand 0 because already in the inventory
            self.best_ligand = unique_ligands[np.argmax(max_yield_per_ligand)]

            self.where_worst = np.array(
                self.data.index[
                    self.data["ligand_smiles"] == self.worst_ligand
                ].tolist()
            )

            self.feauture_labels = {
                "names": {
                    "bases": unique_bases,
                    "ligands": unique_ligands,
                    "aryl_halides": unique_aryl_halides,
                    "additives": unique_additives,
                },
                "ordered_smiles": {
                    "bases": self.data["base_smiles"],
                    "ligands": self.data["ligand_smiles"],
                    "aryl_halides": self.data["aryl_halide_smiles"],
                    "additives": self.data["additive_smiles"],
                },
            }
            
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
                    ligand_price  = row["Ligand_Cost_fixed"]
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




if __name__ == "__main__":

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
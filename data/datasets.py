import numpy as np
import pandas as pd
import torch
import random
from utils import FingerprintGenerator, inchi_to_smiles, convert2pytorch, check_entries
from sklearn.preprocessing import MinMaxScaler
from data.baumgartner import baumgartner
import copy as cp


class Evaluation_data:
    def __init__(
        self, dataset, init_size, prices, init_strategy="values", nucleophile=None
    ):
        self.dataset = dataset
        self.init_strategy = init_strategy
        self.init_size = init_size
        self.prices = prices

        self.ECFP_size = 512
        self.radius = 2

        self.ftzr = FingerprintGenerator(nBits=self.ECFP_size, radius=self.radius)

        if nucleophile is None:
            self.nucleophile = "Aniline"
        else:
            self.nucleophile = nucleophile

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
            self.experiments = cp.deepcopy(self.X)

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

        elif self.dataset == "baumgartner":
            baum = baumgartner(nucleophile=self.nucleophile)
            self.experiments = baum.experiments
            self.X, self.y = baum.X, baum.y

            self.all_precatalysts = baum.all_precatalysts
            self.all_solvents = baum.all_solvents
            self.all_bases = baum.all_bases

            self.worst_precatalyst = baum.worst_precatalyst
            self.where_worst_precatalyst = baum.where_worst_precatalyst

            self.worst_bases = baum.worst_bases
            self.where_worst_bases = baum.where_worst_bases

            self.feauture_labels = baum.feauture_labels

            self.price_dict_precatalyst = baum.price_dict_precatalyst
            self.price_dict_solvent = baum.price_dict_solvent
            self.price_dict_base = baum.price_dict_base

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

        elif self.prices == "update_all_when_bought":
            if self.dataset == "baumgartner":
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

            elif self.dataset == "baumgartner":
                indices_init = np.array(
                    list(
                        set(self.where_worst_precatalyst) & set(self.where_worst_bases)
                    )
                )

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


if __name__ == "__main__":
    DATASET = Evaluation_data(
        "baumgartner",
        200,
        "update_all_when_bought",
        init_strategy="worst_ligand_and_more",
    )

    DATASET.get_init_holdout_data(111)
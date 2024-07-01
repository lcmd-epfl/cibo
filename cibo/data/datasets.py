import numpy as np
import torch
import random
from sklearn.preprocessing import MinMaxScaler

from cibo.utils import convert2pytorch, check_entries
from cibo.data.baumgartner import baumgartner
from cibo.data.directaryl import directaryl
from cibo.data.user_data import user_data


#random.seed(111)
#np.random.seed(111)

class Evaluation_data:
    def __init__(
        self, dataset, init_size, prices, init_strategy="values", nucleophile=None, csv_file=None, description=None
    ):
        self.dataset = dataset
        self.init_strategy = init_strategy
        self.init_size = init_size
        self.prices = prices

        self.csv_file = csv_file
        self.description = description

        if nucleophile is None:
            self.nucleophile = "Aniline"
        else:
            self.nucleophile = nucleophile

        self.get_raw_dataset()

        rep_size = self.X.shape[1]
        self.bounds_norm = torch.tensor([[0] * rep_size, [1] * rep_size])
        self.bounds_norm = self.bounds_norm.to(dtype=torch.float32)

        if not check_entries(self.X):
            self.scaler_X = MinMaxScaler()
            self.X = self.scaler_X.fit_transform(self.X)

    def get_raw_dataset(self):
        # TODO: include this dataset with more reastic prices
        # https://github.com/doyle-lab-ucla/edboplus/blob/main/examples/publication/BMS_yield_cost/data/PCI_PMI_cost_full_update.csv
        # https://chemrxiv.org/engage/chemrxiv/article-details/62f6966269f3a5df46b5584b

        if self.dataset == "BMS":
            # direct arylation reaction
            BMS = directaryl()
            self.data = BMS.data
            self.experiments = BMS.experiments
            self.X, self.y = BMS.X, BMS.y

            self.all_ligands = BMS.all_ligands
            self.all_bases = BMS.all_bases
            self.all_solvents = BMS.all_solvents

            self.best_ligand = BMS.best_ligand
            self.worst_ligand = BMS.worst_ligand
            self.where_worst_ligand = BMS.where_worst_ligand
            self.feauture_labels = BMS.feauture_labels

        elif self.dataset == "user_data":
            if self.description is not None:
                user = user_data(csv_file=self.csv_file, description=self.description)
            else:
                user = user_data(csv_file=self.csv_file)
            self.data = user.data
            self.experiments = user.experiments
            self.X, self.y = user.X, user.y

        elif self.dataset == "baumgartner":
            baum = baumgartner(nucleophile=self.nucleophile)
            self.experiments = baum.experiments
            self.X, self.y = baum.X, baum.y

            self.all_precatalysts = baum.all_precatalysts
            self.all_solvents = baum.all_solvents
            self.all_bases = baum.all_bases

            self.cheapest_precatalyst = baum.cheapest_precatalyst
            self.cheapest_base = baum.cheapest_base
            self.cheapest_solvent = baum.cheapest_solvent
            self.where_cheapest_precatalyst_base_solvent_indices = (
                baum.where_cheapest_precatalyst_base_solvent_indices
            )

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
            if self.dataset == "BMS" or self.dataset == "user_data":
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

        elif self.init_strategy == "random_ligands":
            """
            Randomly select 10 points accross the dataset and set the costs of the corresponding ligands to 0.
            then randomly select 144 points with these ligands as the initial set. all other ligands should still have the original cost.
            """

            chance = np.random.choice(
                np.arange(len(self.y)), size=3, replace=False
            )

            selected_ligands = np.unique(self.all_ligands[chance])

            # from the self.all_ligands, select the indices of the selected ligands
            indices_init = np.array(
                [i for i in np.arange(len(self.y)) if self.all_ligands[i] in selected_ligands]
            )
            # select self.init points random from the indices_init
            indices_init = np.random.choice(
                indices_init, size=self.init_size, replace=False
            )

            costs_init = sum(self.ligand_prices[ligand] for ligand in selected_ligands)
            # make the costs of the selected ligands 0 but to add the cost later on to all methods (BO, CIBO, etc.)
            for ligand in selected_ligands:
                self.ligand_prices[ligand] = 0

            costs_holdout = sum(self.ligand_prices[ligand] for ligand in np.unique(self.all_ligands))
            LIGANDS_INIT = self.all_ligands[indices_init]
            LIGANDS_HOLDOUT = self.all_ligands

            X_init, y_init = self.X[indices_init], self.y[indices_init]
            indices_holdout = np.setdiff1d(np.arange(len(self.y)), indices_init)
            X_holdout, y_holdout = self.X[indices_holdout], self.y[indices_holdout]
            exp_init = self.experiments[indices_init]
            exp_holdout = self.experiments[indices_holdout]

            X_init, y_init = convert2pytorch(X_init, y_init)
            X_holdout, y_holdout = convert2pytorch(X_holdout, y_holdout)
            
            return (X_init, 
                    y_init, 
                    costs_init, 
                    X_holdout, 
                    y_holdout, 
                    costs_holdout, 
                    LIGANDS_INIT,
                    LIGANDS_HOLDOUT,
                    self.ligand_prices,
                    exp_init, 
                    exp_holdout)

        elif self.init_strategy == "worst_ligand":
            indices_init = self.where_worst_ligand[: self.init_size]
            exp_init = self.experiments[indices_init]
            indices_holdout = np.setdiff1d(np.arange(len(self.y)), indices_init)

            np.random.shuffle(indices_init)
            np.random.shuffle(indices_holdout)

            X_init, y_init = self.X[indices_init], self.y[indices_init]
            X_holdout, y_holdout = self.X[indices_holdout], self.y[indices_holdout]
            exp_holdout = self.experiments[indices_holdout]

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
                exp_init,
                exp_holdout,
            )

        elif self.init_strategy == "worst_ligand_and_more":
            if self.dataset == "BMS" or self.dataset == "user_data":
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
                exp_init = self.experiments[indices_init]
                exp_holdout = self.experiments[indices_holdout]

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
                    exp_init,
                    exp_holdout,
                )

        elif self.init_strategy == "cheapest":
            if self.dataset == "baumgartner":
                indices_init = self.where_cheapest_precatalyst_base_solvent_indices
                indices_holdout = np.setdiff1d(np.arange(len(self.y)), indices_init)

                np.random.shuffle(indices_holdout)

                experiments_init = self.experiments[indices_init]
                experiments_holdout = self.experiments[indices_holdout]

                X_init, y_init = self.X[indices_init], self.y[indices_init]
                X_holdout, y_holdout = self.X[indices_holdout], self.y[indices_holdout]

                self.price_dict_precatalyst[self.cheapest_precatalyst] = 0
                self.price_dict_base[self.cheapest_base] = 0
                self.price_dict_solvent[self.cheapest_solvent] = 0

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
                    experiments_init,
                    experiments_holdout,
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

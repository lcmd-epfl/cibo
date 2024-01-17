import pandas as pd
from utils import FingerprintGenerator
import numpy as np
import copy as cp
import pdb


class baumgartner:
    def __init__(self, nucleophile):
        self.data, self.all_price_dicts = self.preprocess()
        self.nucleophile = nucleophile

        self.ECFP_size = 256
        self.radius = 2
        self.ftzr = FingerprintGenerator(nBits=self.ECFP_size, radius=self.radius)

        self.data = self.data[self.nucleophile]
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        data_copy = self.data.copy()
        data_copy.drop("yield", axis=1, inplace=True)
        # check for duplicates
        duplicates = data_copy.duplicated().any()

        if duplicates:
            print("There are duplicates in the dataset.")
            exit()

        self.cheapest_precatalyst_base = self.data[
            (self.data["Precatalyst"] == "tBuXPhos") & (self.data["Base"] == "DBU")
        ]

        if self.nucleophile == "Morpholine":
            # for Morpholine, no experiments with tBuXPhos
            self.cheapest_precatalyst_base = self.data[
                (self.data["Precatalyst"] == "tBuBrettPhos")
                & (self.data["Base"] == "DBU")
            ]

        self.cheapest_precatalyst = np.unique(
            self.cheapest_precatalyst_base.precatalyst_smiles.values
        )[0]

        self.cheapest_base = np.unique(
            self.cheapest_precatalyst_base.base_smiles.values
        )[0]

        self.cheapest_solvent = np.unique(
            self.cheapest_precatalyst_base.solvent_smiles
        )[0]

        self.where_cheapest_precatalyst_base_solvent_indices = self.data[
            (self.data["precatalyst_smiles"] == self.cheapest_precatalyst)
            & (self.data["base_smiles"] == self.cheapest_base)
            & (self.data["solvent_smiles"] == self.cheapest_solvent)
        ].index.to_numpy()

        col_0_precatalyst = self.ftzr.featurize(self.data["precatalyst_smiles"])
        col_1_solvent = self.ftzr.featurize(self.data["solvent_smiles"])
        col_2_base = self.ftzr.featurize(self.data["base_smiles"])
        col_8_base_conc = self.data["Base concentration (M)"].values
        col_10_temp = self.data["Temperature (degC)"].values
        col_11_base_eq = self.data["Base equivalents"].values
        col_12_residence_time = self.data["Residence Time Actual (s)"].values

        self.X = np.concatenate(
            [
                col_0_precatalyst,
                col_1_solvent,
                col_2_base,
                col_8_base_conc.reshape(-1, 1),
                col_10_temp.reshape(-1, 1),
                col_11_base_eq.reshape(-1, 1),
                col_12_residence_time.reshape(-1, 1),
            ],
            axis=1,
        )

        self.experiments = np.concatenate(
            [
                self.data["precatalyst_smiles"].to_numpy().reshape(-1, 1),
                self.data["solvent_smiles"].to_numpy().reshape(-1, 1),
                self.data["base_smiles"].to_numpy().reshape(-1, 1),
                self.data["Base concentration (M)"].to_numpy().reshape(-1, 1),
                self.data["Temperature (degC)"].to_numpy().reshape(-1, 1),
                self.data["Base equivalents"].to_numpy().reshape(-1, 1),
                self.data["Residence Time Actual (s)"].to_numpy().reshape(-1, 1),
                self.data["yield"].to_numpy().reshape(-1, 1),
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
                    self.data[self.data["precatalyst_smiles"] == unique_precatalyst][
                        "yield"
                    ]
                )
                for unique_precatalyst in unique_precatalysts
            ]
        )

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

    def preprocess(self):
        price_data = pd.read_csv(
            "https://raw.githubusercontent.com/janweinreich/rules_of_acquisition/main/data/baumgartner2019_compound_info.csv"
        )

        price_dict_precatalyst = {}
        price_dict_solvent = {}
        price_dict_base = {}

        # select all additives from  the price_data
        precatalyst = price_data[price_data.type == "precatalyst"]
        solvent = price_data[price_data.type == "make-up solvent"]
        base = price_data[price_data.type == "base"]

        for smiles, cost in zip(precatalyst.smiles, precatalyst.cost_per_gram):
            price_dict_precatalyst[smiles] = cost

        for smiles, cost in zip(solvent.smiles, solvent.cost_per_gram):
            price_dict_solvent[smiles] = cost

        for smiles, cost in zip(base.smiles, base.cost_per_gram):
            price_dict_base[smiles] = cost

        all_price_dicts = {
            "precatalyst": price_dict_precatalyst,
            "solvent": price_dict_solvent,
            "base": price_dict_base,
        }

        name_smiles_dict = {}
        for name, smiles in zip(price_data.name, price_data.smiles):
            name_smiles_dict[name] = smiles

        data = pd.read_csv(
            "https://raw.githubusercontent.com/janweinreich/rules_of_acquisition/main/data/baumgartner2019_reaction_data.csv"
        )
        # Unique N-H nucleophiles
        # round yield above 100 to 100
        data["yield"] = data["Reaction Yield"].apply(
            lambda x: 100.0 if x > 100.0 else x
        )

        precatalyst_smiles = []
        solvent_smiles = []
        base_smiles = []
        for precatalyst, solvent, base in zip(
            data["Precatalyst"], data["Make-Up Solvent ID"], data["Base"]
        ):
            precatalyst_smiles.append(name_smiles_dict[precatalyst])
            solvent_smiles.append(name_smiles_dict[solvent])
            base_smiles.append(name_smiles_dict[base])

        data["precatalyst_smiles"] = precatalyst_smiles
        data["solvent_smiles"] = solvent_smiles
        data["base_smiles"] = base_smiles

        nucleophiles = ["Aniline", "Benzamide", "Morpholine", "Phenethylamine"]
        # Create a dictionary of dataframes, each containing data for a specific nucleophile
        dataframes = {}
        for nucleophile in nucleophiles:
            dataframes[nucleophile] = data[data["N-H nucleophile "] == nucleophile]

        return dataframes, all_price_dicts


if __name__ == "__main__":
    # print(baumgartner2019_prices())
    pass

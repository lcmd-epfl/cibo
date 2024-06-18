import numpy as np
import pandas as pd
from cibo.utils import FingerprintGenerator
import pdb


class user_data:
    def __init__(self, csv_file="user_data.csv"):
        # direct arylation reaction
        self.ECFP_size = 512
        self.radius = 2
        self.ftzr = FingerprintGenerator(nBits=self.ECFP_size, radius=self.radius)

        self.data = pd.read_csv(csv_file)
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
        self.experiments = np.concatenate(
            [
                self.data["Base_SMILES"].to_numpy().reshape(-1, 1),
                self.data["Ligand_SMILES"].to_numpy().reshape(-1, 1),
                self.data["Solvent_SMILES"].to_numpy().reshape(-1, 1),
                self.data["Concentration"].to_numpy().reshape(-1, 1),
                self.data["Temp_C"].to_numpy().reshape(-1, 1),
                self.data["Yield"].to_numpy().reshape(-1, 1),
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
            self.data.index[self.data["Ligand_SMILES"] == self.worst_ligand].tolist()
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


class user_data2:

    def __init__(
        self,
        csv_file="user_data.csv",
        description={
            "compounds": {
                "1": {"name": "Ligand_SMILES", "inp_type": "smiles"},
                "2": {"name": "Base_SMILES", "inp_type": "smiles"},
                "3": {"name": "Solvent_SMILES", "inp_type": "smiles"},
            },
            "parameters": {
                "1": {"name": "Concentration", "inp_type": "float"},
                "2": {"name": "Temp_C", "inp_type": "float"},
            },
            "cost": {"name": "Ligand_Cost_fixed", "inp_type": "float"},
            "target": {"name": "Yield", "inp_type": "float"},
        },
    ):
        # direct arylation reaction
        self.description = description
        self.ECFP_size = 512
        self.radius = 2
        self.ftzr = FingerprintGenerator(nBits=self.ECFP_size, radius=self.radius)

        self.data = pd.read_csv(csv_file)

        self.ligand_smiles_name = self.description["compounds"]["1"]["name"]
        self.target_name = self.description["target"]["name"]
        self.X = self.construct_representation()

        self.experiments = self.construct_experiments()
        self.y = self.data[self.target_name].to_numpy()
        self.worst_ligand = self.find_worst_ligand()

    def construct_representation(self):
        rep_vector = []
        compounds = self.description["compounds"]

        # Iterate over the compounds
        for compound in compounds.values():
            compound_name = compound["name"]
            rep_vector.append(self.ftzr.featurize(self.data[compound_name]))

        # Check if parameters are present
        if "parameters" in self.description:
            parameters = self.description["parameters"]
            for parameter in parameters.values():
                parameter_name = parameter["name"]
                rep_vector.append(self.data[parameter_name].to_numpy().reshape(-1, 1))

        # Concatenate all vectors along the second axis
        return np.concatenate(rep_vector, axis=1)

    def construct_experiments(self):
        exp_vector = []
        # Add compounds
        compounds = self.description["compounds"]
        for compound in compounds.values():
            compound_name = compound["name"]
            exp_vector.append(self.data[compound_name].to_numpy().reshape(-1, 1))

        # Add parameters
        if "parameters" in self.description:
            parameters = self.description["parameters"]
            for parameter in parameters.values():
                parameter_name = parameter["name"]
                exp_vector.append(self.data[parameter_name].to_numpy().reshape(-1, 1))

        # Add target
        exp_vector.append(self.data[self.target_name].to_numpy().reshape(-1, 1))

        # Concatenate all vectors along the second axis
        return np.concatenate(exp_vector, axis=1)

    def find_worst_ligand(self):
        unique_ligands = np.unique(self.data[self.ligand_smiles_name])

        max_yield_per_ligand = np.array(
            [
                max(
                    self.data[self.data[self.ligand_smiles_name] == unique_ligand][
                        self.target_name
                    ]
                )
                for unique_ligand in unique_ligands
            ]
        )

        worst_ligand = unique_ligands[np.argmin(max_yield_per_ligand)]
        return worst_ligand


if __name__ == "__main__":
    user_data()

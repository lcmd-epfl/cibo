import numpy as np
import pandas as pd
from cibo.utils import FingerprintGenerator



class user_data:

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



        # check if "cost" is present in the description
        if "cost" in self.description:
            self.data["Ligand_Cost_fixed"] = self.data[self.description["cost"]["name"]]


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
    print("Testing user_data.py")
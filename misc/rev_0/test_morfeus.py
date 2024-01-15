from joblib import Parallel, delayed
import morfeus
from morfeus.conformer import ConformerEnsemble
from morfeus import SASA, Dispersion
import numpy as np
import pdb
import tqdm


def compute_descriptors_from_smiles(
    smiles, normalize=False, missingVal=None, silent=True
):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # List of descriptors to exclude
    exclude_descriptors = [
        "BCUT2D_MWHI",
        "BCUT2D_MWLOW",
        "BCUT2D_CHGHI",
        "BCUT2D_CHGLO",
        "BCUT2D_LOGPHI",
        "BCUT2D_LOGPLOW",
        "BCUT2D_MRHI",
        "BCUT2D_MRLOW",
    ]

    # Get a list of all descriptor calculation functions
    descriptor_names = [
        x[0] for x in Descriptors._descList if x[0] not in exclude_descriptors
    ]

    descriptor_values = []
    nan_descriptors = []

    for name in descriptor_names:
        try:
            descriptor_func = getattr(Descriptors, name)
            value = descriptor_func(mol)

            if math.isnan(value):
                nan_descriptors.append(name)

            descriptor_values.append(value)
        except Exception as e:
            if not silent:
                print(f"Failed to compute descriptor {name}: {e}")
            descriptor_values.append(missingVal)

    if nan_descriptors:
        print(f"Descriptors that returned nan: {nan_descriptors}")

    # Count elements in the molecule
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    element_counts = Counter(atoms)

    # Convert list to a numpy array (vector)
    descriptor_vector = np.array(descriptor_values)

    # Convert element counts to a numpy array and append to the descriptor vector
    element_vector = np.array(
        [
            element_counts.get(element, 0)
            for element in [
                "C",
                "O",
                "N",
                "H",
                "S",
                "F",
                "Cl",
                "Br",
                "I",
                "K",
                "P",
                "Cs",
            ]
        ]
    )
    descriptor_vector = np.concatenate([descriptor_vector, element_vector])

    if normalize:
        # Normalize the vector
        norm = np.linalg.norm(descriptor_vector)
        if norm == 0:
            return descriptor_vector
        return descriptor_vector / norm

    return descriptor_vector


def compute_descriptors_from_smiles_list(
    smiles_list, normalize=False, missingVal=None, silent=True
):
    descriptor_vectors = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            descriptor_vectors.append(None)
            continue

        # List of descriptors to exclude
        exclude_descriptors = [
            "BCUT2D_MWHI",
            "BCUT2D_MWLOW",
            "BCUT2D_CHGHI",
            "BCUT2D_CHGLO",
            "BCUT2D_LOGPHI",
            "BCUT2D_LOGPLOW",
            "BCUT2D_MRHI",
            "BCUT2D_MRLOW",
            "MaxPartialCharge",
            "MinPartialCharge",
            "MaxAbsPartialCharge",
            "MinAbsPartialCharge",
        ]

        # Get a list of all descriptor calculation functions
        descriptor_names = [
            x[0] for x in Descriptors._descList if x[0] not in exclude_descriptors
        ]

        descriptor_values = []
        nan_descriptors = []

        for name in descriptor_names:
            try:
                descriptor_func = getattr(Descriptors, name)
                value = descriptor_func(mol)

                if math.isnan(value):
                    nan_descriptors.append(name)

                descriptor_values.append(value)
            except Exception as e:
                if not silent:
                    print(f"Failed to compute descriptor {name}: {e}")
                descriptor_values.append(missingVal)

        if nan_descriptors:
            print(
                f"Descriptors that returned nan for SMILES {smiles}: {nan_descriptors}"
            )

        # Count elements in the molecule
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        element_counts = Counter(atoms)

        # Convert list to a numpy array (vector)
        descriptor_vector = np.array(descriptor_values)

        # Convert element counts to a numpy array and append to the descriptor vector
        element_vector = np.array(
            [
                element_counts.get(element, 0)
                for element in [
                    "C",
                    "O",
                    "N",
                    "H",
                    "S",
                    "F",
                    "Cl",
                    "Br",
                    "I",
                    "K",
                    "P",
                    "Cs",
                ]
            ]
        )
        descriptor_vector = np.concatenate([descriptor_vector, element_vector])

        if normalize:
            # Normalize the vector
            norm = np.linalg.norm(descriptor_vector)
            if norm == 0:
                descriptor_vector = descriptor_vector
            else:
                descriptor_vector = descriptor_vector / norm

        descriptor_vectors.append(descriptor_vector)

    return np.array(descriptor_vectors)

"""
import leruli

import morfeus
from morfeus.conformer import ConformerEnsemble
from morfeus import SASA, Dispersion

            if MORFEUS:
                self.X = np.concatenate(
                    [
                        col_0_base,
                        col_1_ligand,
                        col_2_aryl_halide,
                        col_3_additive,
                        base_morfeus,
                        ligand_morfeus,
                        aryl_halide_morfeus,
                        additive_morfeus,
                    ],
                    axis=1,
                )

            if MORFEUS:
                morfeus_reps = {
                    "base_smiles": {
                        base: get_morfeus_desc(base) for base in unique_bases
                    },
                    "ligand_smiles": {
                        ligand: get_morfeus_desc(ligand) for ligand in unique_ligands
                    },
                    "aryl_halide_smiles": {
                        aryl_halide: get_morfeus_desc(aryl_halide)
                        for aryl_halide in unique_aryl_halides
                    },
                    "additive_smiles": {
                        additive: get_morfeus_desc(additive)
                        for additive in unique_additives
                    },
                }

                base_morfeus = np.array(
                    [morfeus_reps["base_smiles"][base] for base in data["base_smiles"]]
                )
                ligand_morfeus = np.array(
                    [
                        morfeus_reps["ligand_smiles"][ligand]
                        for ligand in data["ligand_smiles"]
                    ]
                )
                aryl_halide_morfeus = np.array(
                    [
                        morfeus_reps["aryl_halide_smiles"][aryl_halide]
                        for aryl_halide in data["aryl_halide_smiles"]
                    ]
                )
                additive_morfeus = np.array(
                    [
                        morfeus_reps["additive_smiles"][additive]
                        for additive in data["additive_smiles"]
                    ]
                )



def fragments(smiles):
    #"""
    ##auxiliary function to calculate the fragment representation of a molecule
   # """
    # descList[115:] contains fragment-based features only
    # (https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html)
    # Update: in the new RDKit version the indices are [124:]
    fragments = {d[0]: d[1] for d in Descriptors.descList[124:]}
    frags = np.zeros((len(smiles), len(fragments)))
    for i in range(len(smiles)):
        mol = Chem.MolFromSmiles(smiles[i])
        try:
            features = [fragments[d](mol) for d in fragments]
        except:
            raise Exception("molecule {}".format(i) + " is not canonicalised")
        frags[i, :] = features

    return frags

def get_dummy_nuclear_charge_values(smiles_list):
    values = []
    non_none_indices = []

    for index, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            value = dummy_function_nuclear_charge(mol)
            values.append(value)
            non_none_indices.append(index)
        except:
            values.append(None)

    return values, non_none_indices

def dummy_function_nuclear_charge(mol, desired_size=10, width=5):
    """Calculate a simplified value based on nuclear charges of the molecule."""

    total_value = 0
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()  # Get nuclear charge (atomic number)
        total_value += z  # Simply sum up the nuclear charges

    # Adjust for molecule size
    num_atoms = mol.GetNumAtoms()
    size_penalty = 1 if num_atoms == desired_size else 0  # Simple size penalty
    total_value *= size_penalty

    return total_value                

def get_morfeus_desc(smiles):
    ce = ConformerEnsemble.from_rdkit(smiles)

    ce.prune_rmsd()

    ce.sort()

    for conformer in ce:
        sasa = SASA(ce.elements, conformer.coordinates)
        disp = Dispersion(ce.elements, conformer.coordinates)
        conformer.properties["sasa"] = sasa.area
        conformer.properties["p_int"] = disp.p_int
        conformer.properties["p_min"] = disp.p_min
        conformer.properties["p_max"] = disp.p_max

    ce.get_properties()
    a = ce.boltzmann_statistic("sasa")
    b = ce.boltzmann_statistic("p_int")
    c = ce.boltzmann_statistic("p_min")
    d = ce.boltzmann_statistic("p_max")

    return np.array([a, b, c, d])


def morfeus_ftzr(smiles_list):
    return np.array([get_morfeus_desc(smiles) for smiles in smiles_list])
"""


def get_solv_en(smiles):
    try:
        value = leruli.graph_to_solvation_energy(
            smiles, solventname="water", temperatures=[300]
        )["solvation_energies"]["300.0"]
        sleep(0.5)
        return value
    except:
        return None


def get_solv_en_list(smiles_list):
    solv_energies = []
    non_none_indices = []

    for index, smiles in enumerate(smiles_list):
        try:
            value = leruli.graph_to_solvation_energy(
                smiles, solventname="water", temperatures=[300]
            )["solvation_energies"]["300.0"]
            sleep(0.03)
            solv_energies.append(value)
            non_none_indices.append(index)
        except:
            solv_energies.append(None)

    return solv_energies, non_none_indices


def get_MolLogP_list(smiles_list):
    logP_values = []
    non_none_indices = []

    for index, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            logP = Chem.Crippen.MolLogP(mol)
            # add zero centred gaussian noise with std 0.1
            logP += np.random.normal(0, 1.5)
            logP_values.append(logP)
            non_none_indices.append(index)
        except:
            logP_values.append(None)

    return logP_values, non_none_indices


def get_morfeus_desc(smiles):
    ce = ConformerEnsemble.from_rdkit(smiles)

    ce.prune_rmsd()

    ce.sort()

    for conformer in ce:
        sasa = SASA(ce.elements, conformer.coordinates)
        disp = Dispersion(ce.elements, conformer.coordinates)
        conformer.properties["sasa"] = sasa.area
        conformer.properties["p_int"] = disp.p_int
        conformer.properties["p_min"] = disp.p_min
        conformer.properties["p_max"] = disp.p_max

    ce.get_properties()
    a = ce.boltzmann_statistic("sasa")
    b = ce.boltzmann_statistic("p_int")
    c = ce.boltzmann_statistic("p_min")
    d = ce.boltzmann_statistic("p_max")

    return np.array([a, b, c, d])


dataset_url = (
    "https://raw.githubusercontent.com/doylelab/rxnpredict/master/data_table.csv"
)
# load url directly into pandas dataframe
import pandas as pd

data = pd.read_csv(
    dataset_url
)  # .fillna({"base_smiles":"","ligand_smiles":"","aryl_halide_number":0,"aryl_halide_smiles":"","additive_number":0, "additive_smiles": ""}, inplace=False)
# remove rows with nan
data = data.dropna()


ligands = np.unique(data["ligand_smiles"].values)


def get_morfeus_desc_for_ligand(ligand):
    try:
        des = get_morfeus_desc(ligand)
        return des
    except Exception as e:
        return np.nan


results = Parallel(n_jobs=4)(
    delayed(get_morfeus_desc_for_ligand)(ligand) for ligand in ligands
)

# Now 'results' will hold the descriptors for each ligand, or an exception message if something went wrong.
results = np.array(results)

pdb.set_trace()

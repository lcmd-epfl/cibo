import os
import requests
import pandas as pd
import pdb
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import numpy as np
import math
from collections import Counter
import leruli
from time import sleep
import matplotlib.pyplot as plt
import random
import torch
import copy as cp
import morfeus
from morfeus.conformer import ConformerEnsemble
from morfeus import SASA, Dispersion
import deepchem as dc
import pandas as pd
import pickle

def check_entries(array_of_arrays):
    for array in array_of_arrays:
        for item in array:
            if item < 0 or item > 1:
                return False
    return True


def convert2pytorch(X, y):
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float().reshape(-1, 1)
    return X, y


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

  return np.array([a,b,c,d])


def morfeus_ftzr(smiles_list):
    return np.array([get_morfeus_desc(smiles) for smiles in smiles_list])


class Evaluation_data:
    def __init__(self, dataset, init_size, prices, init_strategy="values"):

        self.dataset = dataset
        self.init_strategy = init_strategy
        self.init_size = init_size
        self.prices = prices

        self.ECFP_size = 64
        self.radius = 4

        self.ftzr = dc.feat.CircularFingerprint(size=self.ECFP_size, radius=self.radius)
                    #=dc.feat.RDKitDescriptors() 

        self.get_raw_dataset()

        rep_size = self.X.shape[1]
        self.bounds_norm = torch.tensor([[0]*rep_size, [1]*rep_size])
        self.bounds_norm = self.bounds_norm.to(dtype=torch.float32)

        if not check_entries(self.X):
            print("###############################################")
            print("Entries of X are not between 0 and 1. Adding MinMaxScaler to the pipeline.")
            print("###############################################")
            from sklearn.preprocessing import MinMaxScaler
            self.scaler_X = MinMaxScaler()
            self.X = self.scaler_X.fit_transform(self.X)

        self.get_prices()

    def get_raw_dataset(self):
        
        if self.dataset == "freesolv":

            _, datasets, _ = dc.molnet.load_sampl(featurizer=self.ftzr, splitter='random', transformers = [])
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

        elif self.dataset == "ebdo_direct_arylation":
            
            dataset_url = "https://raw.githubusercontent.com/b-shields/edbo/master/experiments/data/direct_arylation/experiment_index.csv"
            data = pd.read_csv(dataset_url)
            data = data.dropna()
            data = data.sample(frac=1).reset_index(drop=True)
            col_0_base = self.ftzr(data["Base_SMILES"])
            col_1_ligand = self.ftzr(data["Ligand_SMILES"])
            col_2_solvent = self.ftzr(data["Solvent_SMILES"])
            col_3_concentration = data["Concentration"].to_numpy().reshape(-1,1)
            col_4_temperature = data["Temp_C"].to_numpy().reshape(-1,1)

            self.X = np.concatenate([col_0_base,
                                    col_1_ligand,
                                    col_2_solvent,
                                    col_3_concentration,
                                    col_4_temperature,
                                    ],axis=1)


            self.y = data["yield"].to_numpy()            
            
        elif self.dataset == "buchwald":
            MORFEUS = False
            dataset_url = "https://raw.githubusercontent.com/doylelab/rxnpredict/master/data_table.csv"
            #load url directly into pandas dataframe
            
            data = pd.read_csv(dataset_url) #.fillna({"base_smiles":"","ligand_smiles":"","aryl_halide_number":0,"aryl_halide_smiles":"","additive_number":0, "additive_smiles": ""}, inplace=False)
            # remove rows with nan
            data = data.dropna()
            #randomly shuffly df
            data = data.sample(frac=1).reset_index(drop=True)
            unique_bases = data["base_smiles"].unique()
            unique_ligands = data["ligand_smiles"].unique()
            unique_aryl_halides = data["aryl_halide_smiles"].unique()
            unique_additives = data["additive_smiles"].unique()

            if MORFEUS:
                morfeus_reps = {
                                "base_smiles" : {base : get_morfeus_desc(base) for base in unique_bases},
                                "ligand_smiles" : {ligand : get_morfeus_desc(ligand) for ligand in unique_ligands},
                                "aryl_halide_smiles" : {aryl_halide : get_morfeus_desc(aryl_halide) for aryl_halide in unique_aryl_halides},
                                "additive_smiles" : {additive : get_morfeus_desc(additive) for additive in unique_additives}
                                }
                
                base_morfeus = np.array([morfeus_reps["base_smiles"][base] for base in data["base_smiles"]])
                ligand_morfeus = np.array([morfeus_reps["ligand_smiles"][ligand] for ligand in data["ligand_smiles"]])
                aryl_halide_morfeus = np.array([morfeus_reps["aryl_halide_smiles"][aryl_halide] for aryl_halide in data["aryl_halide_smiles"]])
                additive_morfeus = np.array([morfeus_reps["additive_smiles"][additive] for additive in data["additive_smiles"]])

            col_0_base = np.array([self.ftzr.featurize(x)[0] for x in data["base_smiles"]])
            col_1_ligand = np.array([self.ftzr.featurize(x)[0] for x in data["ligand_smiles"]])
            col_2_aryl_halide = np.array([self.ftzr.featurize(x)[0]  for x in data["aryl_halide_smiles"]])
            col_3_additive = np.array([self.ftzr.featurize(x)[0] for x in data["additive_smiles"]])



            self.feauture_labels = {"names": {
                                            "bases":unique_bases,
                                            "ligands":unique_ligands,
                                            "aryl_halides":unique_aryl_halides,
                                            "additives":unique_additives
                                            }
                                    ,
                                    "ordered_smiles": 
                                     {"bases": data["base_smiles"],
                                      "ligands": data["ligand_smiles"],
                                      "aryl_halides": data["aryl_halide_smiles"],
                                      "additives": data["additive_smiles"]
                                     }
                                    }


            if MORFEUS:
                self.X = np.concatenate([col_0_base,
                                        col_1_ligand,
                                        col_2_aryl_halide,
                                        col_3_additive,
                                        base_morfeus,
                                        ligand_morfeus,
                                        aryl_halide_morfeus,
                                        additive_morfeus],axis=1)
            else:
                self.X = np.concatenate([col_0_base,
                                         col_1_ligand,
                                         col_2_aryl_halide,
                                         col_3_additive
                                        ],axis=1)

            self.y = data["yield"].to_numpy()   
            
            
        else:
            print("Dataset not implemented.")
            exit()
        
    def get_prices(self):
        if self.prices == "random":
            self.costs = np.random.randint(2, size=(len(self.X), 1))

        elif self.prices == "update_when_used":
            if self.dataset == "freesolv":
                print("Not implemented.")
                exit()
            elif self.dataset == "ebdo_direct_arylation":
                print("Not implemented.")
                exit()
            elif self.dataset == "buchwald":
                self.ligand_prices = {}
                for ind, unique_ligand in enumerate(self.feauture_labels["names"]["ligands"]):
                    self.ligand_prices[unique_ligand] =   ind+1
 
                
                all_ligand_prices = []
                for ligand in self.feauture_labels["ordered_smiles"]["ligands"]:
                    all_ligand_prices.append(self.ligand_prices[ligand])
                self.costs = np.array(all_ligand_prices).reshape(-1,1)

                pdb.set_trace()



        


        else:
            print("Price model not implemented.")
            exit()

    def get_init_holdout_data(self, SEED):
        random.seed(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        if self.init_strategy == "values":

            """
            Select the init_size worst values and the rest randomly.
            """

            min_val = np.mean(self.y) - 0.2 * abs(np.std(self.y))
            print("min_val", min_val)
            
            index_worst = np.random.choice(np.argwhere(self.y < min_val).flatten(), size=self.init_size, replace=False)
            index_others = np.setdiff1d(np.arange(len(self.y)), index_worst)
            #randomly shuffle the data
            index_others = np.random.permutation(index_others)

            X_init, y_init, costs_init = self.X[index_worst], self.y[index_worst], self.costs[index_worst]
            X_holdout, y_holdout, costs_holdout = self.X[
                index_others], self.y[index_others], self.costs[index_others]

            X_init, y_init       = convert2pytorch(X_init, y_init)
            X_holdout, y_holdout = convert2pytorch(X_holdout, y_holdout)

            return X_init, y_init, costs_init, X_holdout, y_holdout, costs_holdout
                
        elif self.init_strategy == "random":
            """
            Randomly select init_size values.
            """
            indices_init      =  np.random.choice(np.arange(len(self.y)), size=self.init_size, replace=False)
            indices_holdout   =  np.array([i for i in np.arange(len(self.y)) if i not in indices_init])
            
            np.random.shuffle(indices_init)
            np.random.shuffle(indices_holdout)

            X_init, y_init = self.X[indices_init], self.y[indices_init]
            X_holdout, y_holdout = self.X[indices_holdout], self.y[indices_holdout]
            costs_init = self.costs[indices_init]
            costs_holdout = self.costs[indices_holdout]

            X_init, y_init       = convert2pytorch(X_init, y_init)
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
            indices_init = np.argsort(distances)[::-1][:self.init_size]

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

            X_init, y_init       = convert2pytorch(X_init, y_init)
            X_holdout, y_holdout = convert2pytorch(X_holdout, y_holdout)

            return X_init, y_init, costs_init, X_holdout, y_holdout, costs_holdout

        elif self.init_strategy == "ratio_furthest":

            y_sorted = np.sort(self.y)
            cutoff_20_percent = y_sorted[int(0.3 * len(self.y))]

            # Step 2: Identify the indices for the top 20% y values
            indices_top_20 = np.where(self.y >= cutoff_20_percent)[0]

            # Step 3: Retrieve the vectors in X corresponding to these indices
            X_top_20 = self.X[indices_top_20]

            # Step 4: Compute the distance for each row in X against the top 20% y values
            # Using Euclidean distance and taking the minimum distance as an example
            distances = np.min([np.linalg.norm(self.X - x, axis=1)
                            for x in X_top_20], axis=0)

            # Step 5: Sort these distances and get the indices of the 100 largest distances
            indices_init = np.argsort(distances)[::-1][:self.init_size]

            # Step 6: Get the 100 entries corresponding to these indices from X and y
            X_init = self.X[indices_init]
            y_init = self.y[indices_init]
            costs_init = self.costs[indices_init]

            # Step 7: Get the remaining entries
            indices_holdout = np.setdiff1d(np.arange(self.X.shape[0]), indices_init)
            np.random.shuffle(indices_holdout)
            X_holdout = self.X[indices_holdout]
            y_holdout = self.y[indices_holdout]
            costs_holdout = self.costs[indices_holdout]

            # Convert to PyTorch (assuming convert2pytorch is a function you have)
            X_init, y_init = convert2pytorch(X_init, y_init)
            X_holdout, y_holdout = convert2pytorch(X_holdout, y_holdout)

            return X_init, y_init, costs_init, X_holdout, y_holdout, costs_holdout


        else:
            print("Init strategy not implemented.")
            exit()



def plot_utility_BO_vs_RS(y_better_BO_ALL, y_better_RANDOM_ALL, name="./figures/costs.png"):
    """
    Plot the utility of the BO vs RS (Random Search) for each iteration.
    """  
    
    y_BO_MEAN, y_BO_STD = np.mean(y_better_BO_ALL, axis=0), np.std(y_better_BO_ALL, axis=0)
    y_RANDOM_MEAN, y_RANDOM_STD = np.mean(y_better_RANDOM_ALL, axis=0), np.std(y_better_RANDOM_ALL, axis=0)

    lower_rnd = y_RANDOM_MEAN - y_BO_STD
    upper_rnd = y_RANDOM_MEAN + y_BO_STD
    lower_bo = y_BO_MEAN - y_BO_STD
    upper_bo = y_BO_MEAN + y_BO_STD


    NITER = len(y_BO_MEAN)
    fig1, ax1 = plt.subplots()

    

    ax1.plot(np.arange(NITER), y_RANDOM_MEAN, label='Random')
    ax1.fill_between(np.arange(NITER), lower_rnd, upper_rnd, alpha=0.2)
    ax1.plot(np.arange(NITER), y_BO_MEAN, label='Acquisition Function')
    ax1.fill_between(np.arange(NITER), lower_bo, upper_bo, alpha=0.2)
    ax1.set_xlabel('Number of Iterations')
    ax1.set_ylabel('Best Objective Value')
    plt.legend(loc="lower right")
    plt.xticks(list(np.arange(NITER)))
    plt.savefig(name)

    plt.clf()


def plot_costs_BO_vs_RS(running_costs_BO_ALL, running_costs_RANDOM_ALL, name="./figures/costs.png"):
    """
    Plot the running costs of the BO vs RS (Random Search) for each iteration.
    """

    running_costs_BO_ALL_MEAN, running_costs_BO_ALL_STD = np.mean(running_costs_BO_ALL, axis=0), np.std(running_costs_BO_ALL, axis=0)
    running_costs_RANDOM_ALL_MEAN, running_costs_RANDOM_ALL_STD = np.mean(running_costs_RANDOM_ALL, axis=0), np.std(running_costs_RANDOM_ALL, axis=0)
    lower_rnd_costs = running_costs_RANDOM_ALL_MEAN - running_costs_RANDOM_ALL_STD
    upper_rnd_costs = running_costs_RANDOM_ALL_MEAN + running_costs_RANDOM_ALL_STD
    lower_bo_costs = running_costs_BO_ALL_MEAN - running_costs_BO_ALL_STD
    upper_bo_costs = running_costs_BO_ALL_MEAN + running_costs_BO_ALL_STD


    fig2, ax2 = plt.subplots()
    NITER = len(running_costs_BO_ALL_MEAN)

    ax2.plot(np.arange(NITER), running_costs_RANDOM_ALL_MEAN, label='Random')
    ax2.fill_between(np.arange(NITER), lower_rnd_costs, upper_rnd_costs, alpha=0.2)
    ax2.plot(np.arange(NITER), running_costs_BO_ALL_MEAN, label='Acquisition Function')
    ax2.fill_between(np.arange(NITER), lower_bo_costs, upper_bo_costs, alpha=0.2)
    ax2.set_xlabel('Number of Iterations')
    ax2.set_ylabel('Running Costs [$]')
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

def compute_descriptors_from_smiles(smiles, normalize=False, missingVal=None, silent=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # List of descriptors to exclude
    exclude_descriptors = ['BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 
                           'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW']
    
    # Get a list of all descriptor calculation functions
    descriptor_names = [x[0] for x in Descriptors._descList if x[0] not in exclude_descriptors]
    
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
    element_vector = np.array([element_counts.get(element, 0) for element in ['C', 'O', 'N', 'H', 'S', 'F', 'Cl', 'Br', 'I', 'K', 'P', 'Cs']])
    descriptor_vector = np.concatenate([descriptor_vector, element_vector])
    
    if normalize:
        # Normalize the vector
        norm = np.linalg.norm(descriptor_vector)
        if norm == 0: 
            return descriptor_vector
        return descriptor_vector / norm
    
    return descriptor_vector


def compute_descriptors_from_smiles_list(smiles_list, normalize=False, missingVal=None, silent=True):
    descriptor_vectors = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            descriptor_vectors.append(None)
            continue
        
        # List of descriptors to exclude
        exclude_descriptors = ['BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 
                               'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW',
                               'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge']
        
        # Get a list of all descriptor calculation functions
        descriptor_names = [x[0] for x in Descriptors._descList if x[0] not in exclude_descriptors]
        
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
            print(f"Descriptors that returned nan for SMILES {smiles}: {nan_descriptors}")
        
        # Count elements in the molecule
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        element_counts = Counter(atoms)
        
        # Convert list to a numpy array (vector)
        descriptor_vector = np.array(descriptor_values)
        
        # Convert element counts to a numpy array and append to the descriptor vector
        element_vector = np.array([element_counts.get(element, 0) for element in ['C', 'O', 'N', 'H', 'S', 'F', 'Cl', 'Br', 'I', 'K', 'P', 'Cs']])
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

def generate_fingerprints(smiles_list, nBits=512):
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=nBits)
            fp_array = np.array(list(fp.ToBitString()), dtype=int)  # Convert to NumPy array
            fingerprints.append(fp_array)
        else:
            print(f"Could not generate a molecule from SMILES: {smiles}")
            fingerprints.append(np.array([None]))
    return np.array(fingerprints)



def get_solv_en(smiles):
    try:
        value = leruli.graph_to_solvation_energy(smiles, solventname="water", temperatures=[300])["solvation_energies"]["300.0"]
        sleep(0.5)
        return value
    except:
        return None
    

def get_solv_en_list(smiles_list):
    solv_energies = []
    non_none_indices = []
    
    for index, smiles in enumerate(smiles_list):
        try:
            value = leruli.graph_to_solvation_energy(smiles, solventname="water", temperatures=[300])["solvation_energies"]["300.0"]
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
            #add zero centred gaussian noise with std 0.1
            logP += np.random.normal(0, 1.5)
            logP_values.append(logP)
            non_none_indices.append(index)
        except:
            logP_values.append(None)
    
    return logP_values, non_none_indices


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



def fragments(smiles):
    """
    auxiliary function to calculate the fragment representation of a molecule
    """
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


def reaching_max_n(y_better):
    return np.argmax(np.mean(np.array(y_better), axis=0))



#savepkl file
def save_pkl(file, name):
    with open(name, 'wb') as f:
        pickle.dump(file, f)

#load pkl file
def load_pkl(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
#https://codeocean.com/capsule/7056009/tree/v1

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

def check_entries(array_of_arrays):
    for array in array_of_arrays:
        for item in array:
            if item < 0 or item > 1:
                return False
    return True

class Evaluation_data:
    def __init__(self, dataset, init_size, prices, init_strategy="values"):
        self.dataset = dataset
        self.init_strategy = init_strategy
        self.init_size = init_size
        self.prices = prices

        self.get_raw_dataset()

        if not check_entries(self.X):
            print("Entries of X are not between 0 and 1. Add MinMaxScaler to the pipeline.")
            exit()

        self.get_prices()

    def get_raw_dataset(self):
        
        if self.dataset == "freesolv":
            try:
                import deepchem as dc
            except:
                print("Deepchem not installed. Please install deepchem to use this dataset.")
                exit()

            featurizer = dc.feat.CircularFingerprint(size=1024)
            _, datasets, transformers = dc.molnet.load_sampl(featurizer=featurizer, splitter='random', transformers = [])
            train_dataset, valid_dataset, test_dataset = datasets

            X_train = train_dataset.X
            y_train = train_dataset.y[:, 0]
            X_valid = valid_dataset.X
            y_valid = valid_dataset.y[:, 0]
            X_test = test_dataset.X
            y_test = test_dataset.y[:, 0]

            X = np.concatenate((X_train, X_valid, X_test))
            y = np.concatenate((y_train, y_valid, y_test)) 

            random_inds = np.random.permutation(len(X))
            self.X = X[random_inds]
            self.y = y[random_inds]

        elif self.dataset == "buchwald_hartwig":
            pass
        else:
            print("Dataset not implemented.")
            exit()
        
    def get_prices(self):
        if self.prices == "random":
            self.costs = np.random.randint(2, size=(642, 1))
        else:
            print("Price model not implemented.")
            exit()

    def get_init_holdout_data(self):
        if self.init_strategy == "values":

            """
            Select the init_size worst values and the rest randomly.
            """

            indices_worst  =  np.argsort(self.y)[:self.init_size]
            indices_others = np.setdiff1d(np.arange(len(self.y)), indices_worst)
            index_others   = np.random.permutation(index_others)

            X_init, y_init = self.X[index_worst], self.y[index_worst]
            costs_init = self.costs[index_worst]
            costs_candidate = costs[index_others]
            X_candidate, y_candidate = self.X[index_others], self.y[index_others]


            X_init = torch.from_numpy(X_init).float()
            X_candidate = torch.from_numpy(X_candidate).float()
            y_init = torch.from_numpy(y_init).float().reshape(-1,1)
            y_candidate = torch.from_numpy(y_candidate).float().reshape(-1,1)

            return X_init, y_init, costs_init, X_candidate, y_candidate, costs_candidate

        elif self.init_strategy == "random":
            """
            Randomly select init_size values.
            """
            pass
        elif self.init_strategy == "furthest":
            """
            Select molecules furthest away the global optimum.
            """
            pass

        else:
            print("Init strategy not implemented.")
            exit()

        return X_init, y_init, costs_init


def plot_utility_BO_vs_RS(y_better_BO_ALL, y_better_RANDOM_ALL):
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
    plt.savefig("optimization.png")

    plt.clf()


def plot_costs_BO_vs_RS(running_costs_BO_ALL, running_costs_RANDOM_ALL):
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
    ax2.fill_between(np.arange(NITER+1), lower_bo_costs, upper_bo_costs, alpha=0.2)
    ax2.set_xlabel('Number of Iterations')
    ax2.set_ylabel('Running Costs [$]')
    plt.legend(loc="lower right")
    plt.xticks(list(np.arange(NITER)))
    plt.savefig("costs.png")

    plt.clf()


def select_random_smiles(file_path, N, exclude_set=None):
    """
    Select N random SMILES from a gzipped file.
    
    Parameters:
        file_path (str): Path to the gzipped SMILES file.
        N (int): Number of random SMILES to select.
        exclude_set (set): Set of SMILES to exclude from selection.
        
    Returns:
        list: List of N randomly selected SMILES.
    """
    all_smiles = []
    
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            smiles = line.strip().split(' ')[0]  # Take only the first part before the space
            if exclude_set and smiles in exclude_set:
                continue
            all_smiles.append(smiles)
    
    if N > len(all_smiles):
        raise ValueError("N is greater than the number of available SMILES.")
    
    return random.sample(all_smiles, N)



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
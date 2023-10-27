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
import gzip
import random
from process import *

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
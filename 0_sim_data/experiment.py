import numpy as np
import pdb
import deepchem as dc
from process import *
from selection_methods import *
from BO import *
import random  
import matplotlib.pyplot as plt


random.seed(45577)
np.random.seed(4565777)


#https://zinc20.docking.org/substances/subsets/for-sale/


if __name__ == "__main__":

    molecules = select_random_smiles("/home/jan/Downloads/6_p2.smi.gz", 40000, exclude_set=None)
    

    RANDOM_RESULTS, BO_RESULTS = [],[]
    for ITER in range(15):
        random.seed(ITER+1)
        #randon shuffle molecules
        np.random.shuffle(molecules)
        
        N_inital = 100 #30
        
        #select first 10 molecules for training and rest for test set
        initial_molecules           = np.array(canonicalize_smiles_list(molecules[:N_inital]))
        y_initial, no_none          = get_MolLogP_list(initial_molecules)
        initial_molecules           = initial_molecules[no_none]
        y_initial                   = np.array(y_initial)[no_none]
        test_molecules              = np.array(canonicalize_smiles_list(molecules[N_inital:]))
        X_initial                   = compute_descriptors_from_smiles_list(initial_molecules) #, normalize=True)
        X_test                      = compute_descriptors_from_smiles_list(test_molecules) #, normalize = True)
        
        random_experiment = RandomExperiment(y_initial,test_molecules,costly_fct=get_MolLogP_list, n_exp=10, batch_size=20)
        best_molecule_random,y_better_random = random_experiment.run()
        experiment          =   Experiment(X_initial,X_test, y_initial, test_molecules, get_MolLogP_list, acqfct=ExpectedImprovement, n_exp=10, batch_size=20)
        
        best_molecule_BO,y_better_BO = experiment.run()

        RANDOM_RESULTS.append(y_better_random)
        BO_RESULTS.append(y_better_BO)


    RANDOM_RESULTS, BO_RESULTS           = np.array(RANDOM_RESULTS), np.array(BO_RESULTS)
    MEAN_RANDOM_RESULTS, MEAN_BO_RESULTS = np.mean(RANDOM_RESULTS, axis=0),  np.mean(BO_RESULTS, axis=0)
    STD_RANDOM_RESULTS, STD_BO_RESULTS   = np.std(RANDOM_RESULTS, axis=0),  np.std(BO_RESULTS, axis=0)
    


    lower_rnd = MEAN_RANDOM_RESULTS - STD_RANDOM_RESULTS
    upper_rnd = MEAN_RANDOM_RESULTS + STD_RANDOM_RESULTS
    lower_ei = MEAN_BO_RESULTS - STD_BO_RESULTS
    upper_ei = MEAN_BO_RESULTS + STD_BO_RESULTS



    iters = np.arange(len(MEAN_RANDOM_RESULTS))
    plt.plot(iters, MEAN_RANDOM_RESULTS, label='Random')
    plt.fill_between(iters, lower_rnd, upper_rnd, alpha=0.2)
    plt.plot(iters, MEAN_BO_RESULTS, label='EI')
    plt.fill_between(iters, lower_ei, upper_ei, alpha=0.2)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Best Objective Value')
    plt.legend(loc="lower right")
    plt.xticks(list(iters))
    plt.savefig("test.png")
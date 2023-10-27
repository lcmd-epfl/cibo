import numpy as np
import pdb
import deepchem as dc
from process import *
from selection_methods import *
from BO import *
import random  
import matplotlib.pyplot as plt
from plot_selection import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

random.seed(45577)
np.random.seed(4565777)

#https://zinc20.docking.org/substances/subsets/for-sale/

def gen_rep_from_df(rep_type_1="ECFP",rep_type_2="onehot", dimred=False):

    experiment = pd.read_csv("experiment_index.csv")
    #randomly shuffle the data
    experiment = experiment.sample(frac=1)
    BASE_SMILES    = experiment["Base_SMILES"].values
    SOLVENT_SMILES = experiment["Solvent_SMILES"].values
    
    
    CONCENTRATION  = experiment["Concentration"].values
    TEMPERATURE    = experiment["Temp_C"].values
    
    
    LIGAND_SMILES  = experiment["Ligand_SMILES"].values


    if rep_type_2 == "onehot":

        
    
        encoder1,encoder2,  = OneHotEncoder(sparse=False),OneHotEncoder(sparse=False)

        BASE_SMILES_2D = BASE_SMILES.reshape(-1, 1)
        BASE_SMILES_one_hot = encoder1.fit_transform(BASE_SMILES_2D)
        
        SOLVENT_SMILES_2D = SOLVENT_SMILES.reshape(-1, 1)
        SOLVENT_SMILES_one_hot = encoder2.fit_transform(SOLVENT_SMILES_2D)

        X_BASE = BASE_SMILES_one_hot
        X_SOLVENT = SOLVENT_SMILES_one_hot

    elif rep_type_2 == "ECFP":
        X_BASE    = generate_fingerprints(BASE_SMILES, nBits=64)
        X_SOLVENT = generate_fingerprints(SOLVENT_SMILES, nBits=64)

    elif rep_type_2 == "descriptors":
        X_BASE    = compute_descriptors_from_smiles_list(BASE_SMILES)
        X_SOLVENT = compute_descriptors_from_smiles_list(SOLVENT_SMILES)

    elif rep_type_2 == "frags":
        X_BASE    = fragments(BASE_SMILES)
        X_SOLVENT = fragments(SOLVENT_SMILES)

    elif rep_type_2 == "mix":
        X_BASE    = np.column_stack((fragments(BASE_SMILES),  generate_fingerprints(BASE_SMILES, nBits=32)))
        X_SOLVENT = np.column_stack((fragments(SOLVENT_SMILES), generate_fingerprints(SOLVENT_SMILES, nBits=32)))

    else:
        raise ValueError("Unknown representation type")
        

    if rep_type_1 == "ECFP":
        X_LIGAND =  generate_fingerprints(BASE_SMILES, nBits=256)
    elif rep_type_1 == "onehot":
        encoder3 = OneHotEncoder(sparse=False)
        LIGAND_SMILES_2D = LIGAND_SMILES.reshape(-1, 1)
        X_LIGAND = encoder3.fit_transform(LIGAND_SMILES_2D)
    elif rep_type_1 == "descriptors":
        X_LIGAND = compute_descriptors_from_smiles_list(LIGAND_SMILES)
    elif rep_type_1 == "frags":
        X_LIGAND = fragments(LIGAND_SMILES)
    else:
        raise ValueError("Unknown representation type")

    X = np.column_stack((X_BASE, X_LIGAND, X_SOLVENT, CONCENTRATION, TEMPERATURE))

    if dimred:
        X = MinMaxScaler().fit_transform(X)
        umap_model = umap.UMAP(n_components=100)
        umap_model.fit(X)
        X = umap_model.transform(X)

    y = experiment["yield"].values

    return LIGAND_SMILES, X, y

            
def split_for_bo(rep_type_1="ECFP", rep_type_2="onehot", train_size=100, dimred=False):
    LIGAND_SMILES,X,y = gen_rep_from_df(rep_type_1=rep_type_1,rep_type_2=rep_type_2, dimred=dimred)
    
    #get the worst 100 with yield < 70
    index_worst = np.random.choice(np.argwhere(y<70).flatten(), size=train_size, replace=False)
    index_others = np.setdiff1d(np.arange(len(y)), index_worst)
    #randomly shuffle the data
    index_others = np.random.permutation(index_others)
    LIGAND_SMILES_train, LIGAND_SMILES_test, X_train, X_test, y_train, y_test = LIGAND_SMILES[index_worst], LIGAND_SMILES[index_others], X[index_worst], X[index_others], y[index_worst], y[index_others]
    
    return LIGAND_SMILES_train, LIGAND_SMILES_test, X_train, X_test, y_train, y_test
    

FIT_TEST = False
BAYES_OPT = True

if __name__ == "__main__":
    if FIT_TEST:
        #scale the features
        LIGAND_SMILES,X, y = gen_rep_from_df(rep_type="ECFP", dimred=False)
        PLOT_SOAP(MinMaxScaler().fit_transform(X), y, label='yield', dimred="PCA", selection=None, random_points=None)
        

        X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2)
        model = CustomGPModel(kernel_type="Matern")
        model.fit(X_train, y_train)
        y_pred, sigma = model.predict(X_test)

        #make a scatter plot
        
        plt.scatter(y_test, y_pred, c='blue', alpha=0.5, label='Predicted')
        plt.errorbar(y_test, y_pred, yerr=sigma, fmt='o', ecolor='gray', capsize=5)
        plt.xlabel("Experimental")
        plt.ylabel("Predicted")
        plt.savefig("./figures/scatter.png")

        mae = mean_absolute_error(y_test, y_pred)
        #rmse
        rmse = np.mean((y_test - y_pred)**2)**0.5
        #r2
        r2 = r2_score(y_test, y_pred)
        print("MAE: ", mae, "RMSE: ", rmse, "R2: ", r2)

    if BAYES_OPT:    

        all_tests = [{"rep_type_1":"ECFP" ,"rep_type_2": "ECFP", "acqft": "EI" ,"dimred": False}, 
                     {"rep_type_1":"ECFP" ,"rep_type_2": "onehot", "dimred": False},
                     {"rep_type_1":"ECFP" ,"rep_type_2": "descriptors", "dimred": False},
                     {"rep_type_1":"ECFP" ,"rep_type_2": "frags", "dimred": False},
                     {"rep_type_1":"ECFP" ,"rep_type_2": "mix", "dimred": False},
                     {"rep_type_1":"descriptors" ,"rep_type_2": "ECFP", "dimred": False},
                     {"rep_type_1":"descriptors" ,"rep_type_2": "onehot", "dimred": False},
                     {"rep_type_1":"descriptors" ,"rep_type_2": "descriptors", "dimred": False},
                     {"rep_type_1":"descriptors" ,"rep_type_2": "frags", "dimred": False},
                     {"rep_type_1":"descriptors" ,"rep_type_2": "mix", "dimred": False},
                     {"rep_type_1":"frags" ,"rep_type_2": "ECFP", "dimred": False},
                     {"rep_type_1":"frags" ,"rep_type_2": "onehot", "dimred": False},
                     {"rep_type_1":"frags" ,"rep_type_2": "descriptors", "dimred": False},
                     {"rep_type_1":"frags" ,"rep_type_2": "frags", "dimred": False},
                     {"rep_type_1":"frags" ,"rep_type_2": "mix", "dimred": False}]

 
        RANDOM_RESULTS, BO_RESULTS = [],[]
        for ITER in range(15):
            LIGAND_SMILES_train, LIGAND_SMILES_test, X_train,X_test, y_train, y_test = split_for_bo(rep_type_1="ECFP", rep_type_2="ECFP", train_size=300, dimred=False)
            initial_molecules           = LIGAND_SMILES_train
            y_initial                   = y_train
            test_molecules              = LIGAND_SMILES_test
            X_initial                   = X_train

            random_experiment   = RandomExperimentHoldout(y_initial,test_molecules,y_test, n_exp=5, batch_size=3)
            random_experiment.run()
            best_molecule_random,y_better_random = random_experiment.run()             
            RANDOM_RESULTS.append(y_better_random)

        for ITER in range(15):

            LIGAND_SMILES_train, LIGAND_SMILES_test, X_train,X_test, y_train, y_test = split_for_bo(rep_type_1="ECFP", rep_type_2="ECFP", train_size=300, dimred=False)
            initial_molecules           = LIGAND_SMILES_train
            y_initial                   = y_train
            test_molecules              = LIGAND_SMILES_test
            X_initial                   = X_train
                     
            experiment          =   ExperimentHoldout(X_initial,y_initial,test_molecules,X_test,y_test,type_acqfct="EI", n_exp=5, batch_size=3, fps=False)
            best_molecule_BO,y_better_BO = experiment.run()
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
        plt.plot(iters, MEAN_BO_RESULTS, label='Acquisition Function')
        plt.fill_between(iters, lower_ei, upper_ei, alpha=0.2)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Best Objective Value')
        plt.legend(loc="lower right")
        plt.xticks(list(iters))
        plt.savefig("test.png")
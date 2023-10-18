

import numpy as np
import pdb
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from process import *
import torch
import gpytorch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, LinearKernel,PolynomialKernel, AdditiveKernel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

class CustomGPModel:
    def __init__(self, kernel_type="RBF"):
        self.kernel_type = kernel_type
        self.scaler_X = MinMaxScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X_train, y_train):
        X_train = self.scaler_X.fit_transform(X_train)
        self.scaler_y.fit(y_train.reshape(-1, 1))
        y_train = self.scaler_y.transform(y_train.reshape(-1, 1))

        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float64).view(-1, 1)

        if self.kernel_type == "RBF":
            kernel = RBFKernel()
        elif self.kernel_type == "Matern":
            kernel = MaternKernel(nu=2.5)
        elif self.kernel_type == "Linear":
            kernel = LinearKernel()
        elif self.kernel_type == "Polynomial":
            kernel = PolynomialKernel(power=2, offset=0.0)
        elif self.kernel_type == "Product":
            kernel = MaternKernel(nu=2.5,active_dims=torch.tensor(np.arange(216).tolist()))*RBFKernel(active_dims=torch.tensor([216]))*RBFKernel(active_dims=torch.tensor([217]))
        else:
            raise ValueError("Invalid kernel type")

        class InternalGP(SingleTaskGP):
            def __init__(self, train_X, train_Y, kernel):
                super().__init__(train_X, train_Y)
                self.covar_module = ScaleKernel(kernel)
        #If in doubt consult
        #https://github.com/pytorch/botorch/blob/main/tutorials/fit_model_with_torch_optimizer.ipynb
        self.gp = InternalGP(self.X_train_tensor, self.y_train_tensor, kernel)
        self.gp.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.GreaterThan(1e-5))
        self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        self.mll.to(self.X_train_tensor)
        fit_gpytorch_model(self.mll)

    def predict(self, X_test):
        X_test_norm = self.scaler_X.transform(X_test)
        X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float64)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.gp.posterior(X_test_tensor)
            mean = posterior.mean
            std_dev = posterior.variance.sqrt()

        mean_original = self.scaler_y.inverse_transform(mean.detach().numpy().reshape(-1, 1)).flatten()
        std_dev_original = std_dev.detach().numpy() * self.scaler_y.scale_
        std_dev_original = std_dev_original.flatten()

        return mean_original, std_dev_original

def UCB(X_candidates, gp_model, kappa=1.0):
    """
    Calculate the Upper Confidence Bound (UCB) acquisition function values for given candidates.
    
    Parameters:
        X_candidates (numpy.ndarray): The candidate points where the acquisition function should be evaluated.
        gp_model (GaussianProcessRegressor): The trained Gaussian Process model.
        kappa (float): The exploration-exploitation parameter.
        
    Returns:
        numpy.ndarray: The UCB values at the candidate points.
    """
    mu, sigma = gp_model.predict(X_candidates, return_std=True)
    neg_ucb_values = -mu - kappa * sigma
    return neg_ucb_values, mu, sigma

def ExpectedImprovement(X_candidates, gp_model,y_best, kappa=0.01):
    mu, sigma = gp_model.predict(X_candidates)
    
    with np.errstate(divide='warn'):
        Z = (y_best - mu - kappa) / (sigma + 1e-9)
    ei_values = (y_best - mu - kappa) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei_values[sigma <= 1e-9] = 0.0
    return ei_values, mu, sigma


def exploration_estimate(values, p=0.02):
    """
    Estimate the exploration parameter kappa scales sigma values in acquisition functions.
    Parameters:
        values (numpy.ndarray): The values of the objective function.
        p (float): The percentile of the values to use.
    Returns:
        float: The exploration parameter kappa.
    """
    
    return np.mean(np.abs(values)*p)

class RandomExperimentHoldout:
    """
    Larger is better!
    """
    def __init__(self, y_initial,test_molecules,y_test, n_exp=10, batch_size=10):
        self.y_initial = y_initial
        self.test_molecules = test_molecules
        self.y_test = y_test
        self.batch_size = batch_size
        self.n_exp = n_exp

    def run(self):
        y_better = []
        y_best = np.max(self.y_initial)
        y_better.append(y_best)

        best_molecule = None

        for i in range(self.n_exp):
            #take k random unique indices from test molecules
            top_k_indices = np.random.choice(len(self.test_molecules), self.batch_size, replace=False)
            top_k_molecules = self.test_molecules[top_k_indices]
            y_top_k         = self.y_test[top_k_indices]
    
            if y_best < np.max(y_top_k):
                y_best = np.max(y_top_k)
                best_molecule = top_k_molecules[np.argmax(y_top_k)]
                print("New best molecule: ", best_molecule, "New best value: ", y_best)
            
            
            y_top_k = np.array(y_top_k)
            self.test_molecules = np.delete(self.test_molecules, top_k_indices)
            self.y_test = np.delete(self.y_test, top_k_indices)
            y_better.append(y_best)
            print("Iteration: ", i, "Top k values: ", y_top_k, "Best value: ", y_best)
        
        y_better = np.array(y_better)

        return best_molecule,y_better

class ExperimentHoldout:
    """
    Larger is better!
    """
    def __init__(self,X_initial, y_initial,test_molecules,X_test,y_test,acqfct=ExpectedImprovement, n_exp=10, batch_size=10):
        
        self.X_initial = X_initial
        self.y_initial = y_initial
        self.test_molecules = test_molecules
        self.X_test = X_test
        self.y_test = y_test
        self.acqfct = acqfct
        self.batch_size = batch_size
        self.n_exp = n_exp

        self.kernel_type = "RBF"

        model = CustomGPModel(kernel_type=self.kernel_type)
        model.fit(self.X_initial, self.y_initial)
        self.surrogate = model


    def run(self):
        X, y = self.X_initial, self.y_initial
        mu = self.y_initial
        y_better = []
        y_best = np.max(self.y_initial)
        y_better.append(y_best)

        best_molecule = None

        for i in range(self.n_exp):
            
            acqfct_values, mu, sigma = self.acqfct(self.X_test, self.surrogate,y_best, kappa = exploration_estimate(mu, p=0.05))          
            top_k_indices   = np.argsort(acqfct_values)[:self.batch_size]
            top_k_molecules = self.test_molecules[top_k_indices]
            y_top_k         = self.y_test[top_k_indices]
            X_top_k         = self.X_test[top_k_indices]
    
            if y_best < np.max(y_top_k):
                y_best = np.max(y_top_k)
                best_molecule = top_k_molecules[np.argmax(y_top_k)]
                print("New best molecule: ", best_molecule, "New best value: ", y_best)
            
            y_better.append(y_best)
            self.test_molecules = np.delete(self.test_molecules, top_k_indices)
            self.X_test = np.delete(self.X_test, top_k_indices, axis=0)
            self.y_test = np.delete(self.y_test, top_k_indices)
            
            X = np.concatenate((X, X_top_k))
            y = np.concatenate((y, y_top_k))
            model = CustomGPModel(kernel_type=self.kernel_type)
            model.fit(X, y)
            self.surrogate = model
            print("Iteration: ", i, "Top k values: ", y_top_k, "Best value: ", y_best)

        y_better = np.array(y_better)

        return best_molecule,y_better


class RandomExperiment:
    def __init__(self, y_initial,test_molecules,costly_fct, n_exp=10, batch_size=10):
        self.y_initial = y_initial
        self.test_molecules = test_molecules
        self.costly_fct = costly_fct
        self.batch_size = batch_size
        self.n_exp = n_exp

    def run(self):
        y_better = []
        y_best = np.min(self.y_initial)
        y_better.append(y_best)

        best_molecule = None

        for i in range(self.n_exp):
            #take k random unique indices from test molecules
            top_k_indices = np.random.choice(len(self.test_molecules), self.batch_size, replace=False)
            top_k_molecules = self.test_molecules[top_k_indices]
            y_top_k, no_none = self.costly_fct(top_k_molecules)
    
            if y_best > np.min(y_top_k):
                y_best = np.min(y_top_k)
                best_molecule = top_k_molecules[np.argmin(y_top_k)]
                print("New best molecule: ", best_molecule, "New best value: ", y_best)
            
            top_k_molecules = top_k_molecules[no_none]
            y_top_k = np.array(y_top_k)[no_none]
            self.test_molecules = np.delete(self.test_molecules, top_k_indices)
            y_better.append(y_best)
            print("Iteration: ", i, "Top k values: ", y_top_k, "Best value: ", y_best)

        y_better = np.array(y_better)

        return best_molecule,y_better


def LogNoisyExpectedImprovement(X_candidates, gp_model, n_fantasies=10):

    """

    Calculate the Log Noisy Expected Improvement (LogNEI) at given points.
    Parameters:

        X_candidates (ndarray): Points at which to evaluate the LogNEI
        gp_model (GaussianProcessRegressor): Trained GP model
        y_best (float): The best observed value
        n_fantasies (int): The number of fantasy samples to average over

    Returns:

        lognei_values (ndarray): LogNEI values at X_candidates

    """
    mu, sigma = gp_model.predict(X_candidates, return_std=True)
    # Generate fantasy samples
    fantasy_samples = np.random.normal(loc=mu, scale=sigma, size=(n_fantasies, len(X_candidates)))
    # Calculate EI for each fantasy sample
    ei_fantasy_values = []
    for fantasy in fantasy_samples:
        fantasy_best = np.max(fantasy)
        with np.errstate(divide='warn'):
            Z = (fantasy_best - mu) / (sigma + 1e-9)
        ei_values = np.maximum(fantasy_best - mu, 0) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei_values[sigma <= 1e-9] = 0.0
        ei_fantasy_values.append(ei_values)


    # Average EI over all fantasy samples and take the log
    avg_ei_values = np.mean(ei_fantasy_values, axis=0)
    lognei_values = np.log(avg_ei_values + 1e-9)  # Add epsilon to avoid log(0)

    return lognei_values, mu, sigma


class Experiment:
    def __init__(self, X_initial,X_test, y_initial,test_molecules,costly_fct,acqfct=ExpectedImprovement,  n_exp=10, batch_size=10):

        """
        Runs the experiment.
        Smaller values are better.

        Parameters:
            X_initial (numpy.ndarray): The initial molecules (descriptors).
            y_initial (numpy.ndarray): The initial values.
            test_molecules (numpy.ndarray): The test molecules (SMILES).
            costly_fct (function): The function that returns the values for a list of molecules.
            acqfct (function): The acquisition function to use.
            n_exp (int): The number of experiments to run.
            batch_size (int): The number of molecules to evaluate in each iteration.
        """

        self.X_initial = X_initial
        self.y_initial = y_initial
        self.test_molecules = test_molecules
        self.X_test    = X_test
        self.costly_fct = costly_fct

        self.acqfct = acqfct
        self.batch_size = batch_size
        self.n_exp = n_exp


        model = CustomGPModel(kernel_type="Matern")
        model.fit(self.X_initial, self.y_initial)
        self.surrogate = model

    def run(self):
        X, y = self.X_initial, self.y_initial
        mu = self.y_initial

        y_better = []#
        y_best = np.min(self.y_initial)
        y_better.append(y_best)

        best_molecule = None
        for i in range(self.n_exp):
            
            acqfct_values, mu, sigma = self.acqfct(self.X_test, self.surrogate,y_best, kappa = exploration_estimate(mu, p=0.2))          
            top_k_indices = np.argsort(acqfct_values)[::-1][:self.batch_size]  #np.argsort(acqfct_values)[::-1][:self.batch_size]
            top_k_molecules = self.test_molecules[top_k_indices]
            y_top_k, no_none = self.costly_fct(top_k_molecules)
    
            if y_best > np.min(y_top_k):
                y_best = np.min(y_top_k)
                best_molecule = top_k_molecules[np.argmin(y_top_k)]
                print("New best molecule: ", best_molecule, "New best value: ", y_best)
            
            top_k_molecules = top_k_molecules[no_none]
            y_top_k = np.array(y_top_k)[no_none]
            self.test_molecules = np.delete(self.test_molecules, top_k_indices)
            self.X_test = np.delete(self.X_test, top_k_indices, axis=0)

            y_better.append(y_best)
            X_top_k = compute_descriptors_from_smiles_list(top_k_molecules)
            
            X = np.concatenate((X, X_top_k))
            y = np.concatenate((y, y_top_k))
            model = CustomGPModel(kernel_type="Matern")
            model.fit(X, y)
            self.surrogate = model
            print("Iteration: ", i, "Top k values: ", y_top_k, "Best value: ", y_best)

        y_better = np.array(y_better)

        return best_molecule,y_better



if __name__ == "__main__":
    import deepchem as dc
    featurizer = dc.feat.RDKitDescriptors()


    # Load FreeSolv dataset
    tasks, datasets, transformers = dc.molnet.load_sampl(featurizer=featurizer, splitter='random', transformers = [])
    train_dataset, valid_dataset, test_dataset = datasets

    # Extract training data from DeepChem dataset
    X_train = train_dataset.X
    y_train = train_dataset.y[:, 0]



    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    #normalize the data with standard scaler
    model = CustomGPModel(kernel_type="Matern")
    model.fit(X_train, y_train)
    y_pred, sigma = model.predict(X_test)

    #make a scatter plot
    import matplotlib.pyplot as plt
    plt.scatter(y_test, y_pred, c='blue', alpha=0.5, label='Predicted')
    plt.errorbar(y_test, y_pred, yerr=sigma, fmt='o', ecolor='gray', capsize=5)
    plt.xlabel("Experimental")
    plt.ylabel("Predicted")
    plt.show()
    pdb.set_trace()
    #compute mae
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test, y_pred)
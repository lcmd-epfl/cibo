import numpy as np
from numba import jit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import warnings

@jit(nopython=True)
def rbf_kernel(x1, x2, length_scale, signal_variance):
    return signal_variance * np.exp(-0.5 * np.sum((x1 - x2)**2) / length_scale**2)

@jit(nopython=True)
def rbf_kernel_matrix(X1, X2, length_scale, signal_variance):
    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = rbf_kernel(X1[i], X2[j], length_scale, signal_variance)
    return K

@jit(nopython=True)
def log_likelihood(params, X, y):
    length_scale, signal_variance, noise_variance = params
    N = X.shape[0]
    K = rbf_kernel_matrix(X, X, length_scale, signal_variance) + (noise_variance + 1e-2) * np.eye(N)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    return -0.5 * np.dot(y.T, alpha) - np.sum(np.log(np.diag(L))) - 0.5 * N * np.log(2 * np.pi)

@jit(nopython=True)
def update(params, X, y, lr=0.01):
    epsilon = 1e-3
    grads = np.zeros_like(params)
    for i in range(len(params)):
        params_copy = params.copy()
        params_copy[i] += epsilon
        grads[i] = (log_likelihood(params_copy, X, y) - log_likelihood(params, X, y)) / epsilon
    new_params = params + lr * grads
    new_params = np.clip(new_params, 1e-3, 1e5)
    return new_params

class GaussianProcessRegressor:
    def __init__(self):
        self.params = None
        self.L = None
        self.alpha = None
        self.X = None

    def fit(self, X, y, lr=0.0001, n_iter=100):
        self.X = X
        length_scale = 1.0
        signal_variance = 1.0
        noise_variance = 0.1
        self.params = np.array([length_scale, signal_variance, noise_variance])

        for i in range(n_iter):
            self.params = update(self.params, self.X, y, lr=lr)

        K = rbf_kernel_matrix(self.X, self.X, self.params[0], self.params[1]) + (self.params[2] + 1e-6) * np.eye(self.X.shape[0])
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
        print(self.params)

    def predict(self, X_test, return_std=False):
        K_s = rbf_kernel_matrix(X_test, self.X, self.params[0], self.params[1])
        K_ss_diag = np.diag(rbf_kernel_matrix(X_test, X_test, self.params[0], self.params[1]))
        mu = np.dot(K_s, self.alpha)

        v = np.linalg.solve(self.L, K_s.T)
        var_diag = K_ss_diag - np.sum(v**2, axis=0)
        var_diag = np.maximum(var_diag, 0.0)  # Ensure that variance is non-negative
        std_dev = np.sqrt(var_diag)


        if return_std:
            return mu, std_dev
        else:
            return mu




class EnsembleRegressor:
    def __init__(self):
        self.models = [
            ("Ridge", Ridge()),
            ("Lasso", Lasso()),
            ("Support Vector Regression", SVR()),
            ("Random Forest", RandomForestRegressor()),
            ("Gradient Boosting", GradientBoostingRegressor()),
            ("K-Nearest Neighbors", KNeighborsRegressor())
        ]
        self.fitted_models = {}
        self.predictions = {}
        
    def fit(self, X, y):
        for name, model in self.models:
            try:
                model.fit(X, y)
                self.fitted_models[name] = model
            except Exception as e:
                print(f"Could not fit {name} because of {e}")
                
    def predict(self, X):
        all_preds = []
        for name, model in self.fitted_models.items():
            try:
                y_pred = model.predict(X)
                self.predictions[name] = y_pred
                all_preds.append(y_pred)
            except Exception as e:
                print(f"Could not make predictions with {name} because of {e}")

        # Convert to numpy array for easier manipulation
        all_preds = np.array(all_preds)

        # Compute average and standard deviation of predictions
        avg_prediction = np.mean(all_preds, axis=0)
        std_prediction = np.std(all_preds, axis=0)

        return all_preds, avg_prediction, std_prediction

def loocv_with_splits(X, y, num_splits=5):
    """
    Perform LOOCV multiple times with different dataset shuffles.
    Compute the average and standard deviation of individual errors.
    
    Parameters:
    - X: np.array, the feature matrix
    - y: np.array, the target vector
    - num_splits: int, the number of different dataset shuffles
    
    Returns:
    - avg_individual_errors: np.array, the average individual errors
    - std_individual_errors: np.array, the standard deviation of individual errors
    """
    n = len(y)
    all_individual_errors = np.zeros((num_splits, n))
    
    for i in range(num_splits):
        # Shuffle the data
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Initialize LeaveOneOut and LinearRegression
        loo = LeaveOneOut()
        model = LinearRegression()

        # Perform Leave-One-Out Cross-Validation
        for train_index, test_index in loo.split(X_shuffled):
            X_train, X_test = X_shuffled[train_index], X_shuffled[test_index]
            y_train, y_test = y_shuffled[train_index], y_shuffled[test_index]

            # Fit the model
            model.fit(X_train, y_train)

            # Make a prediction
            y_pred = model.predict(X_test)

            # Compute the error for this point
            error = np.abs(y_test - y_pred)

            # Store the error
            original_index = indices[test_index][0]
            all_individual_errors[i, original_index] = error[0]
    
    # Compute the average and standard deviation of individual errors
    avg_individual_errors = np.mean(all_individual_errors, axis=0)
    std_individual_errors = np.std(all_individual_errors, axis=0)
    
    return avg_individual_errors, std_individual_errors



def UCB(X_candidates, gp_model, kappa=0.1):
    """
    Calculate the Upper Confidence Bound (UCB) acquisition function values for given candidates.
    
    Parameters:
        X_candidates (numpy.ndarray): The candidate points where the acquisition function should be evaluated.
        gp_model (GaussianProcessRegressor): The trained Gaussian Process model.
        kappa (float): The exploration-exploitation parameter.
        
    Returns:
        numpy.ndarray: The UCB values at the candidate points.
    """
    mu, sigma = gp_model.predict(X_candidates)
    ucb_values = mu  + kappa * sigma
    return ucb_values, mu, sigma

def ExpectedImprovement(X_candidates, gp_model,y_best, kappa=0.01):
    mu, sigma = gp_model.predict(X_candidates)
    
    with np.errstate(divide='warn'):
        Z = (y_best - mu - kappa) / (sigma + 1e-9)
    ei_values = (y_best - mu - kappa) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei_values[sigma <= 1e-9] = 0.0
    return ei_values, mu, sigma



def fps_selection_simple(X, n_to_select):

    """

    Farthest Point Sampling (FPS) selection function.
    Parameters:

    - X: np.ndarray, the data points [n_samples, n_features]

    - n_to_select: int, the number of points to select
    Returns:
    - selected_idx: list of int, the indices of the selected points
    """

    X = StandardScaler().fit_transform(X)

    n_samples = X.shape[0]
    selected_idx = []

    # Initialize with a random point
    current_idx = np.random.randint(n_samples)
    selected_idx.append(current_idx)

    # Initialize distance vector
    distance = pairwise_distances(X, X[current_idx, :].reshape(1, -1)).flatten()
    # Iteratively add points

    for _ in range(n_to_select - 1):

        farthest_point_idx = np.argmax(distance)
        selected_idx.append(farthest_point_idx)
        # Update distance vector
        new_distance = pairwise_distances(X, X[farthest_point_idx, :].reshape(1, -1)).flatten()
        distance = np.minimum(distance, new_distance)

    return selected_idx


def LogNoisyExpectedImprovement(X_candidates, gp_model,n_fantasies=10):

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
    mu, sigma = gp_model.predict(X_candidates)
    # Generate fantasy samples
    fantasy_samples = np.random.normal(loc=mu, scale=sigma, size=(n_fantasies, len(X_candidates)))
    #np.random.normal(loc=mu, scale=sigma, size=(n_fantasies, len(X_candidates)))
    #you can assume different noise models, e.g. laplace, normal, etc. , laplacian would be 
    # equivalent to having some outliers in the data
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



        
def BatchFantasizingWithEI(gp_model, X, Y, X_candidates, batch_size=3, n_fantasies=10):
    # Initialize variables
    X_batch = []
    inds_batch = []
    Y_fantasies = [cp.deepcopy(Y) for _ in range(n_fantasies)]
    
    # Make a copy of X_candidates to remove selected candidates later
    remaining_candidates = cp.deepcopy(X_candidates)

    # Calculate the initial EI and select x1
    ei_values, _, _ = ExpectedImprovement(remaining_candidates, gp_model, np.max(Y), kappa=0.01)
    top_ind = np.argsort(ei_values)
    x1 = remaining_candidates[top_ind[0]]
    X_batch.append(x1)
    inds_batch.append(top_ind[0])

    # Remove the selected candidate
    remaining_candidates = np.delete(remaining_candidates, top_ind[0], axis=0)

    for j in range(1, batch_size):
        avg_ei = np.zeros(remaining_candidates.shape[0])

        for i in range(n_fantasies):
            # Copy the existing GP model
            dummy_model = cp.deepcopy(gp_model)

            # Update the model with new point and corresponding fantasy y-value
            X_dummy = np.vstack([X, np.array(X_batch)])
            y_dummy = Y_fantasies[i]
            
            mu, sigma = dummy_model.predict(np.array(X_batch))
            sampled_y = np.random.normal(mu[-1], sigma[-1], 1)
            
            y_dummy = np.append(y_dummy, sampled_y)
            Y_fantasies[i] = y_dummy
            
            dummy_model.fit(X_dummy, y_dummy)
            
            ei_fantasy, _, _ = ExpectedImprovement(remaining_candidates, dummy_model, np.max(y_dummy), kappa=0.01)
            avg_ei += ei_fantasy

        avg_ei /= n_fantasies
        
        max_ei_ind = np.argmax(avg_ei)
        xj = remaining_candidates[max_ei_ind]
        X_batch.append(xj)

        # Remove the selected candidate
        remaining_candidates = np.delete(remaining_candidates, max_ei_ind, axis=0)
        
        # Append index of selected candidate
        inds_batch.append(max_ei_ind)
    
    return inds_batch, mu, sigma


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
    def __init__(self,X_initial, y_initial,test_molecules,X_test,y_test,type_acqfct="EI", n_exp=10, batch_size=10, fps=False):
        
        self.X_initial = X_initial
        self.y_initial = y_initial
        self.test_molecules = test_molecules
        self.X_test = X_test
        self.y_test = y_test

        self.type_acqfct = type_acqfct
        self.fps = fps

        if type_acqfct == "EI":
            self.acqfct = ExpectedImprovement
        elif type_acqfct == "UCB":
            self.acqfct = UCB
        elif type_acqfct == "LogNEI":
            self.acqfct = LogNoisyExpectedImprovement
        elif type_acqfct == "CostAwareEI":
            pass
        elif type_acqfct == "BatchFantasizingWithEI":
            self.acqfct = BatchFantasizingWithEI
        else:
            raise ValueError("Invalid acquisition function type")
        self.batch_size = batch_size
        self.n_exp = n_exp

        self.kernel_type = "Matern" #"RBF"

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
            
            #
            if self.type_acqfct == "EI":
                acqfct_values, mu, sigma = self.acqfct(self.X_test, self.surrogate,y_best, kappa = exploration_estimate(mu, p=0.05)) 
                top_k_indices   = np.argsort(acqfct_values)[:self.batch_size]         
            elif self.type_acqfct == "UCB":
                acqfct_values, mu, sigma = self.acqfct(self.X_test, self.surrogate, kappa = exploration_estimate(mu, p=0.05))
                top_k_indices   = np.argsort(acqfct_values)[::-1][:self.batch_size]
            elif self.type_acqfct == "LogNEI":
                acqfct_values, mu, sigma = self.acqfct(self.X_test, self.surrogate,n_fantasies=10)
                top_k_indices   = np.argsort(acqfct_values)[:self.batch_size]
            elif self.type_acqfct == "CostAwareEI":
                pass
            elif self.type_acqfct == "BatchFantasizingWithEI":
                top_k_indices, mu, sigma = self.acqfct(self.surrogate,X,y, self.X_test, batch_size=self.batch_size, n_fantasies=10)
            else:
                raise ValueError("Invalid acquisition function type")
            
            if not self.fps:
                top_k_molecules = self.test_molecules[top_k_indices]
                y_top_k         = self.y_test[top_k_indices]
                X_top_k         = self.X_test[top_k_indices]
            else:
                top_100_indices   = np.argsort(acqfct_values)[:100]
                top_100_molecules = self.test_molecules[top_100_indices]
                top_100_y         = self.y_test[top_100_indices]
                top_100_X         = self.X_test[top_100_indices]

                top_k_indices   = fps_selection_simple(top_100_X, self.batch_size)
                top_k_molecules = top_100_molecules[top_k_indices]
                y_top_k         = top_100_y[top_k_indices]
                X_top_k         = top_100_X[top_k_indices]

            
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

        y_better = []
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


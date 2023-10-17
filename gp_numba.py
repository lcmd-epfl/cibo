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
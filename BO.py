

import numpy as np
import pdb
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from process import *
import torch
import gpytorch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood,LeaveOneOutPseudoLikelihood
from botorch.fit import fit_gpytorch_model,fit_gpytorch_mll
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, LinearKernel,PolynomialKernel, AdditiveKernel
from gpytorch.kernels import Kernel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, entropy
from scipy.optimize import minimize
import copy as cp
from sklearn.metrics import pairwise_distances
from torch.optim import Adam, Adamax
import torch
from torch import Tensor

random.seed(45577)
np.random.seed(4565777)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float
class TensorStandardScaler:
    def __init__(self, dim: int = -2, epsilon: float = 1e-9):
        self.dim = dim
        self.epsilon = epsilon
        self.mean = None
        self.std = None

    def fit(self, Y):
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).float()
        self.mean = Y.mean(dim=self.dim, keepdim=True)
        self.std = Y.std(dim=self.dim, keepdim=True)
        self.std = self.std.where(self.std >= self.epsilon, torch.full_like(self.std, 1.0))

    def transform(self, Y):
        if self.mean is None or self.std is None:
            raise ValueError("Mean and standard deviation not initialized, run `fit` method first.")
        original_type = None
        if isinstance(Y, np.ndarray):
            original_type = np.ndarray
            Y = torch.from_numpy(Y).float()
        Y_transformed = (Y - self.mean) / self.std
        if original_type is np.ndarray:
            return Y_transformed.numpy()
        else:
            return Y_transformed

    def fit_transform(self, Y):
        self.fit(Y)
        return self.transform(Y)

    def inverse_transform(self, Y):
        if self.mean is None or self.std is None:
            raise ValueError("Mean and standard deviation not initialized, run `fit` method first.")
        original_type = None
        if isinstance(Y, np.ndarray):
            original_type = np.ndarray
            Y = torch.from_numpy(Y).float()
        Y_inv_transformed = (Y * self.std) + self.mean
        if original_type is np.ndarray:
            return Y_inv_transformed.numpy()
        else:
            return Y_inv_transformed


#from here https://github.com/leojklarner/gauche/blob/main/gauche/kernels/fingerprint_kernels/tanimoto_kernel.py
def batch_tanimoto_sim(
        x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Tanimoto similarity between two batched tensors, across last 2 dimensions.
    eps argument ensures numerical stability if all zero tensors are added. Tanimoto similarity is proportional to:

    (<x, y>) / (||x||^2 + ||y||^2 - <x, y>)

    where x and y may be bit or count vectors or in set notation:

    |A \cap B | / |A| + |B| - |A \cap B |

    Args:
        x1: `[b x n x d]` Tensor where b is the batch dimension
        x2: `[b x m x d]` Tensor
        eps: Float for numerical stability. Default value is 1e-6
    Returns:
        Tensor denoting the Tanimoto similarity.
    """

    if x1.ndim < 2 or x2.ndim < 2:
        raise ValueError("Tensors must have a batch dimension")

    dot_prod = torch.matmul(x1, torch.transpose(x2, -1, -2))
    x1_norm = torch.sum(x1 ** 2, dim=-1, keepdims=True)
    x2_norm = torch.sum(x2 ** 2, dim=-1, keepdims=True)

    tan_similarity = (dot_prod + eps) / (
            eps + x1_norm + torch.transpose(x2_norm, -1, -2) - dot_prod
    )

    return tan_similarity.clamp_min_(0)  # zero out negative values for numerical stability


class TanimotoKernel(Kernel):
    r"""
     Computes a covariance matrix based on the Tanimoto kernel
     between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

     .. math::

    \begin{equation*}
     k_{\text{Tanimoto}}(\mathbf{x}, \mathbf{x'}) = \frac{\langle\mathbf{x},
     \mathbf{x'}\rangle}{\left\lVert\mathbf{x}\right\rVert^2 + \left\lVert\mathbf{x'}\right\rVert^2 -
     \langle\mathbf{x}, \mathbf{x'}\rangle}
    \end{equation*}

    .. note::

     This kernel does not have an `outputscale` parameter. To add a scaling parameter,
     decorate this kernel with a :class:`gpytorch.test_kernels.ScaleKernel`.

     Example:
         >>> x = torch.randint(0, 2, (10, 5))
         >>> # Non-batch: Simple option
         >>> covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
         >>> covar = covar_module(x)  # Output: LazyTensor of size (10 x 10)
         >>>
         >>> batch_x = torch.randint(0, 2, (2, 10, 5))
         >>> # Batch: Simple option
         >>> covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
         >>> covar = covar_module(batch_x)  # Output: LazyTensor of size (2 x 10 x 10)
    """

    is_stationary = False
    has_lengthscale = False

    def __init__(self, **kwargs):
        super(TanimotoKernel, self).__init__(**kwargs)

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            assert x1.size() == x2.size() and torch.equal(x1, x2)
            return torch.ones(
                *x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device
            )
        else:
            return self.covar_dist(x1, x2, **params)

    def covar_dist(
            self,
            x1,
            x2,
            last_dim_is_batch=False,
            **params,
    ):
        r"""This is a helper method for computing the bit vector similarity between
        all pairs of points in x1 and x2.

        Args:
            :attr:`x1` (Tensor `n x d` or `b1 x ... x bk x n x d`):
                First set of data.
            :attr:`x2` (Tensor `m x d` or `b1 x ... x bk x m x d`):
                Second set of data.
            :attr:`last_dim_is_batch` (tuple, optional):
                Is the last dimension of the data a batch dimension or not?

        Returns:
            (:class:`Tensor`, :class:`Tensor) corresponding to the distance matrix between `x1` and `x2`.
            The shape depends on the kernel's mode
            * `diag=False`
            * `diag=False` and `last_dim_is_batch=True`: (`b x d x n x n`)
            * `diag=True`
            * `diag=True` and `last_dim_is_batch=True`: (`b x d x n`)
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        return batch_tanimoto_sim(x1, x2)

class CustomGPModel:
    def __init__(self, kernel_type="RBF", scale_type_X="sklearn", bounds_norm=None):
        self.kernel_type  = kernel_type
        self.scale_type_X = scale_type_X
        self.bounds_norm  = bounds_norm

        if scale_type_X == "sklearn":
            self.scaler_X = MinMaxScaler()
        elif scale_type_X == "botorch":
            #bounds_norm = torch.tensor([[0]*1024, [1]*1024])
            #bounds_norm = bounds_norm.to(dtype=torch.float32)
            pass
        else:
            raise ValueError("Invalid scaler type")


        self.scaler_y = TensorStandardScaler()
        #StandardScaler()

    def fit(self, X_train, y_train):
        if self.scale_type_X == "sklearn":
            X_train = self.scaler_X.fit_transform(X_train)
        elif self.scale_type_X == "botorch":
            from botorch.utils.transforms import normalize
            
            if type(X_train) == np.ndarray:
                X_train = torch.tensor(X_train, dtype=torch.float32)


            X_train = normalize(X_train, bounds=self.bounds_norm).to(dtype=torch.float32) 
            #(X_train - self.bounds_norm[0]) / (self.bounds_norm[1] - self.bounds_norm[0])
            #
            
        
        
        #self.scaler_y.fit(y_train.reshape(-1, 1))
        y_train = self.scaler_y.fit_transform(y_train)
        # y_train.reshape(-1, 1)
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
        elif self.kernel_type == "Tanimoto":
            kernel = TanimotoKernel()
        else:
            raise ValueError("Invalid kernel type")

        class InternalGP(SingleTaskGP):
            def __init__(self, train_X, train_Y, kernel):
                super().__init__(train_X, train_Y)
                self.covar_module = ScaleKernel(kernel)
        #If in doubt consult
        #https://github.com/pytorch/botorch/blob/main/tutorials/fit_model_with_torch_optimizer.ipynb
        
        self.gp = InternalGP(self.X_train_tensor, self.y_train_tensor, kernel)
        #self.gp.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.GreaterThan(1e-5))


        FIT_METHOD = True
        if FIT_METHOD:
            self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
            self.mll.to(self.X_train_tensor)
            #fit_gpytorch_model(self.mll, max_retries=2000)
            fit_gpytorch_mll(self.mll, num_retries=5000)
            #pdb.set_trace()
        else:
            self.mll = LeaveOneOutPseudoLikelihood(self.gp.likelihood, self.gp)
            self.mll.to(self.X_train_tensor)
            optimizer = Adamax([{"params": self.gp.parameters()}], lr=1e-1)
            self.gp.train()

            LENGTHSCALE_GRID, NOISE_GRID = np.meshgrid(np.logspace(-3, 3, 10), np.logspace(-5, 1, 10))
            NUM_EPOCHS_INIT = 50


            best_loss = float('inf')
            best_lengthscale = None
            best_noise = None

            # Loop over each grid point
            for lengthscale, noise in zip(LENGTHSCALE_GRID.flatten(), NOISE_GRID.flatten()):
                # Manually set the hyperparameters
                self.gp.covar_module.base_kernel.lengthscale = lengthscale
                self.gp.likelihood.noise = noise
                
                # Perform a brief round of training to get a loss value
                for epoch in range(NUM_EPOCHS_INIT):
                    # clear gradients
                    optimizer.zero_grad()
                    # forward pass through the model to obtain the output MultivariateNormal
                    output = self.gp(self.X_train_tensor)
                    # calculate the negative log likelihood
                    loss = -self.mll(output, self.y_train_tensor.flatten())
                    # back prop gradients
                    loss.backward()
                    optimizer.step()
                
                # If this loss is the best so far, update best_loss and best hyperparameters
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_lengthscale = lengthscale
                    best_noise = noise
                #print(f"Finished grid point with lengthscale {lengthscale}, noise {noise}, loss {loss.item()}")


            # Set the best found hyperparameters
            self.gp.covar_module.base_kernel.lengthscale = best_lengthscale
            self.gp.likelihood.noise = best_noise
            print(f"Best initial lengthscale: {best_lengthscale}, Best initial noise: {best_noise}")


            """
            finished hyperparameter optimization
            """


            NUM_EPOCHS = 1000
            
            for epoch in range(NUM_EPOCHS):
                # clear gradients
                optimizer.zero_grad()
                # forward pass through the model to obtain the output MultivariateNormal
                output = self.gp(self.X_train_tensor)
                # calculate the negative log likelihood
                loss = -self.mll(output, self.y_train_tensor.flatten())
                # back prop gradients
                loss.backward()
                # print every 10 iterations
                if (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch+1:>3}/{NUM_EPOCHS} - Loss: {loss.item():>4.3f} "
                        f"lengthscale: {self.gp.covar_module.base_kernel.lengthscale.item():>4.3f} "
                        f"noise: {self.gp.likelihood.noise.item():>4.3f}"
                    )
                optimizer.step()

            self.gp.eval()
    
        return self.gp

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



def find_min_max_distance_and_ratio(x, vectors):
    # Calculate the squared differences between x and each vector in the list
    squared_diffs_x = np.sum((vectors - x)**2, axis=1)
    # Calculate the Euclidean distances between x and each vector in the list
    distances_x = np.sqrt(squared_diffs_x)
    # Calculate all pairwise squared differences among the vectors in the list
    pairwise_diffs = np.sum((vectors[:, np.newaxis, :] - vectors[np.newaxis, :, :])**2, axis=-1)
    # Calculate the pairwise Euclidean distances among the vectors in the list
    pairwise_distances = np.sqrt(pairwise_diffs)
    # Find the minimal distance
    min_distance = np.min(distances_x)
    # Find the maximal distance among all vectors, including x
    max_distance = np.max([np.max(pairwise_distances), np.max(distances_x)])
    # Calculate the ratio p = min_distance / max_distance
    p = min_distance / max_distance
    return min_distance, max_distance, 1- p



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
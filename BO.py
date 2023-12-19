import numpy as np
import torch
import warnings

import gpytorch
from gpytorch.kernels import Kernel
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf_discrete
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, LinearKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam
from botorch_ext import ForestSurrogate
from sklearn.ensemble import RandomForestRegressor

# Custom module imports
from botorch_ext import optimize_acqf_discrete_modified

# specific import for the modified GIBBON function
from utils import (
    function_cost,
    function_cost_B,
)

# Suppress warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}


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
    #from here https://github.com/leojklarner/gauche/blob/main/gauche/kernels/fingerprint_kernels/tanimoto_kernel.py
    """

    if x1.ndim < 2 or x2.ndim < 2:
        raise ValueError("Tensors must have a batch dimension")

    dot_prod = torch.matmul(x1, torch.transpose(x2, -1, -2))
    x1_norm = torch.sum(x1**2, dim=-1, keepdims=True)
    x2_norm = torch.sum(x2**2, dim=-1, keepdims=True)

    tan_similarity = (dot_prod + eps) / (
        eps + x1_norm + torch.transpose(x2_norm, -1, -2) - dot_prod
    )

    return tan_similarity.clamp_min_(
        0
    )  # zero out negative values for numerical stability


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


class BoundedKernel(gpytorch.kernels.Kernel):
    def __init__(self, lower=0, upper=100):
        super().__init__()
        self.base_kernel = gpytorch.kernels.LinearKernel()
        self.lower = lower
        self.upper = upper

    def forward(self, x1, x2, diag=False, **params):
        base_kernel_matrix = self.base_kernel.forward(x1, x2, diag=diag, **params)

        if isinstance(base_kernel_matrix, gpytorch.lazy.LazyTensor):
            base_kernel_matrix = base_kernel_matrix.evaluate()

        bounded_kernel_matrix = self.lower + (self.upper - self.lower) * torch.sigmoid(
            base_kernel_matrix
        )

        if diag:
            # Assuming bounded_kernel_matrix is already the diagonal
            return bounded_kernel_matrix

        return gpytorch.lazy.NonLazyTensor(bounded_kernel_matrix)


def update_model(
    X,
    y,
    bounds_norm,
    kernel_type="Tanimoto",
    fit_y=True,
    FIT_METHOD=True,
    surrogate="GP",
):
    """
    Update and return a Gaussian Process (GP) model with new training data.
    This function configures and optimizes the GP model based on the provided parameters.

    Args:
        X (numpy.ndarray): The training data, typically feature vectors.
        y (numpy.ndarray): The corresponding labels or values for the training data.
        bounds_norm (numpy.ndarray): Normalization bounds for the training data.
        kernel_type (str, optional): Type of kernel to be used in the GP model. Default is "Tanimoto".
        fit_y (bool, optional): Flag to indicate if the output values (y) should be fitted. Default is True.
        FIT_METHOD (bool, optional): Flag to indicate the fitting method to be used. Default is True.
        surrogate (str, optional): Type of surrogate model to be used. Default is "GP".

    Returns:
        model (botorch.models.gpytorch.GP): The updated GP model, fitted with the provided training data.
        scaler_y (TensorStandardScaler): The scaler used for the labels, which can be applied for future data normalization.

    Notes:
        The function initializes a GP model with specified kernel and fitting methods, then fits the model to the provided data.
        The 'bounds_norm' parameter is used for normalizing the training data within the GP model.
        The 'fit_y' and 'FIT_METHOD' parameters control the fitting behavior of the model.
    """

    GP_class = Surrogate_Model(
        kernel_type=kernel_type,
        bounds_norm=bounds_norm,
        fit_y=fit_y,
        FIT_METHOD=FIT_METHOD,
        surrogate=surrogate,
    )
    model = GP_class.fit(X, y)

    return model, GP_class.scaler_y


class TensorStandardScaler:
    """
    StandardScaler for tensors that standardizes features by removing the mean
    and scaling to unit variance, as defined in BoTorch.

    Attributes:
        dim (int): The dimension over which to compute the mean and standard deviation.
        epsilon (float): A small constant to avoid division by zero in case of a zero standard deviation.
        mean (Tensor, optional): The mean value computed in the `fit` method. None until `fit` is called.
        std (Tensor, optional): The standard deviation computed in the `fit` method. None until `fit` is called.

    Args:
        dim (int): The dimension over which to standardize the data. Default is -2.
        epsilon (float): A small constant to avoid division by zero. Default is 1e-9.
    """

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
        self.std = self.std.where(
            self.std >= self.epsilon, torch.full_like(self.std, 1.0)
        )

    def transform(self, Y):
        if self.mean is None or self.std is None:
            raise ValueError(
                "Mean and standard deviation not initialized, run `fit` method first."
            )
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
            raise ValueError(
                "Mean and standard deviation not initialized, run `fit` method first."
            )
        original_type = None
        if isinstance(Y, np.ndarray):
            original_type = np.ndarray
            Y = torch.from_numpy(Y).float()
        Y_inv_transformed = (Y * self.std) + self.mean
        if original_type is np.ndarray:
            return Y_inv_transformed.numpy()
        else:
            return Y_inv_transformed


class Surrogate_Model:
    """
    A surrogate model class that supports different types of kernels and surrogate methods.

    This class encapsulates the functionality to create and train a surrogate model
    used for predictions in Bayesian Optimization and related tasks.

    Attributes:
        kernel_type (str): Type of the kernel to be used in Gaussian Process (GP).
        bounds_norm (np.ndarray, optional): Bounds for normalizing the input data.
        fit_y (bool): Flag indicating whether to fit the output values.
        FIT_METHOD (bool): Indicates the method for fitting the GP hyperparameters.
        surrogate (str): Type of surrogate model to be used ('GP' for Gaussian Process or 'RF' for Random Forest).
        scaler_y (TensorStandardScaler): Scaler for standardizing the output values.

    Args:
        kernel_type (str): Specifies the kernel type for the GP. Default is "Tanimoto".
        bounds_norm (np.ndarray, optional): Normalization bounds for the data. Default is None.
        fit_y (bool): Indicates if the output values should be fitted. Default is True.
        FIT_METHOD (bool): Selects the fitting method. Default is True.
        surrogate (str): Chooses the surrogate model type. Default is "GP".

    Notes:
        The class supports different kernel types for Gaussian Processes and also allows for
        the use of Random Forest as a surrogate model. The choice of kernel and surrogate
        model type can significantly affect the model's performance.
    """

    def __init__(
        self,
        kernel_type="Tanimoto",
        bounds_norm=None,
        fit_y=True,
        FIT_METHOD=True,
        surrogate="GP",
    ):
        self.kernel_type = kernel_type
        self.bounds_norm = bounds_norm
        self.fit_y = fit_y
        self.surrogate = surrogate
        self.FIT_METHOD = FIT_METHOD
        self.scaler_y = TensorStandardScaler()

    def fit(self, X_train, y_train):
        if type(X_train) == np.ndarray:
            X_train = torch.tensor(X_train, dtype=torch.float32)

        if self.fit_y:
            y_train = self.scaler_y.fit_transform(y_train)
        else:
            y_train = y_train

        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float64).view(-1, 1)

        """
        Use BoTorch fit method
        to fit the hyperparameters of the GP and the model weights
        """
        if self.surrogate == "GP":
            if self.kernel_type == "RBF":
                kernel = RBFKernel()
            elif self.kernel_type == "Matern":
                kernel = MaternKernel(nu=2.5)
            elif self.kernel_type == "Linear":
                kernel = LinearKernel()
            elif self.kernel_type == "Tanimoto":
                kernel = TanimotoKernel()

            elif self.kernel_type == "Bounded":
                boundary = np.array([[0], [100]])
                lo = float(self.scaler_y.transform(boundary)[0])
                hi = float(self.scaler_y.transform(boundary)[1])
                kernel = BoundedKernel(lower=lo, upper=hi)
            else:
                raise ValueError("Invalid kernel type")

            class InternalGP(SingleTaskGP):
                def __init__(self, train_X, train_Y, kernel):
                    super().__init__(train_X, train_Y)
                    self.mean_module = ConstantMean()
                    self.covar_module = ScaleKernel(kernel)

            self.gp = InternalGP(self.X_train_tensor, self.y_train_tensor, kernel)
            if self.kernel_type == "Linear" or self.kernel_type == "Tanimoto":
                self.gp.likelihood.noise_constraint = gpytorch.constraints.GreaterThan(
                    1e-3
                )

            self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
            self.mll.to(self.X_train_tensor)
            if self.FIT_METHOD:
                fit_gpytorch_model(self.mll, max_retries=50000)
            else:
                if self.kernel_type == "RBF" or self.kernel_type == "Matern":
                    self.NUM_EPOCHS_GD = 5000
                elif self.kernel_type == "Linear" or self.kernel_type == "Tanimoto":
                    self.NUM_EPOCHS_GD = 1000
                optimizer = Adam([{"params": self.gp.parameters()}], lr=1e-1)
                self.gp.train()
                if self.gp.covar_module.base_kernel.lengthscale != None:
                    LENGTHSCALE_GRID, NOISE_GRID = np.meshgrid(
                        np.logspace(-3, 3, 10), np.logspace(-5, 1, 10)
                    )
                else:
                    NOISE_GRID = np.logspace(-5, 1, 10)
                    LENGTHSCALE_GRID = np.zeros_like(NOISE_GRID)

                NUM_EPOCHS_INIT = 50

                best_loss = float("inf")
                best_lengthscale = None
                best_noise = None

                # Loop over each grid point
                for lengthscale, noise in zip(
                    LENGTHSCALE_GRID.flatten(), NOISE_GRID.flatten()
                ):
                    # Manually set the hyperparameters
                    if self.gp.covar_module.base_kernel.lengthscale != None:
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

                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        if self.gp.covar_module.base_kernel.lengthscale != None:
                            best_lengthscale = lengthscale
                        best_noise = noise

                if self.gp.covar_module.base_kernel.lengthscale != None:
                    self.gp.covar_module.base_kernel.lengthscale = best_lengthscale
                self.gp.likelihood.noise = best_noise
                if self.gp.covar_module.base_kernel.lengthscale != None:
                    print(
                        f"Best initial lengthscale: {best_lengthscale}, Best initial noise: {best_noise}"
                    )
                else:
                    print(f"Best initial noise: {best_noise}")

                for epoch in range(self.NUM_EPOCHS_GD):
                    # clear gradients
                    optimizer.zero_grad()
                    # forward pass through the model to obtain the output MultivariateNormal
                    output = self.gp(self.X_train_tensor)
                    loss = -self.mll(output, self.y_train_tensor.flatten())
                    loss.backward()
                    if (epoch + 1) % 100 == 0:
                        if self.gp.covar_module.base_kernel.lengthscale != None:
                            print(
                                f"Epoch {epoch+1:>3}/{self.NUM_EPOCHS_GD} - Loss: {loss.item():>4.3f} "
                                f"lengthscale: {self.gp.covar_module.base_kernel.lengthscale.item():>4.3f} "
                                f"noise: {self.gp.likelihood.noise.item():>4.3f}"
                            )
                        else:
                            print(
                                f"Epoch {epoch+1:>3}/{self.NUM_EPOCHS_GD} - Loss: {loss.item():>4.3f} "
                                f"noise: {self.gp.likelihood.noise.item():>4.3f}"
                            )
                    optimizer.step()

            self.gp.eval()
            self.mll.eval()

            return self.gp

        elif self.surrogate == "RF":
            model = RandomForestRegressor(
                n_estimators=100, max_depth=20, random_state=42
            )

            model.fit(self.X_train_tensor, self.y_train_tensor)
            self.model = ForestSurrogate(model)
            return self.model

        else:
            raise ValueError("Invalid surrogate type")


def find_indices(X_candidate_BO, candidates):
    """
    Identifies and returns the indices of specific candidates within a larger dataset.
    This function is particularly useful when the order of candidates returned by an
    acquisition function differs from the original dataset order.

    Args:
        X_candidate_BO (numpy.ndarray): The complete dataset or holdout set,
            typically consisting of feature vectors.
        candidates (numpy.ndarray): A subset of the dataset (e.g., a batch of
            molecules) selected by the acquisition function.

    Returns:
        numpy.ndarray: An array of indices corresponding to the positions of
            each candidate in the original dataset 'X_candidate_BO'.
    """

    indices = []
    for candidate in candidates:
        indices.append(np.argwhere((X_candidate_BO == candidate).all(1)).flatten()[0])
    indices = np.array(indices)
    return indices


def opt_acqfct(
    X_train,
    model,
    X_candidate_BO,
    bounds_norm,
    q,
    sequential=False,
    maximize=True,
    n_best=300,
    acq_func="GIBBON",
):
    """
    Optimizes an acquisition function for Bayesian Optimization.

    This function selects the best candidates from a given set based on the specified acquisition function.
    It supports different types of acquisition functions like 'GIBBON' and 'NEI'.

    Args:
        X_train (np.ndarray): The training data used in the model.
        model (GP model): The Gaussian Process model fitted on the training data.
        X_candidate_BO (np.ndarray): The set of candidate points for selection.
        bounds_norm (np.ndarray): Normalization bounds for the data.
        q (int): The number of candidates to select in each iteration.
        sequential (bool, optional): If True, performs sequential optimization. Default is False.
        maximize (bool, optional): Indicates if the goal is to maximize the acquisition function. Default is True.
        n_best (int, optional): The number of best candidates to return. Default is 300.
        acq_func (str, optional): The type of acquisition function to use. Default is "GIBBON".

    Returns:
        tuple:
            - index_set (np.ndarray): Indices of the selected candidates in the original candidate set.
            - acq_values (np.ndarray): Acquisition values of the selected candidates.
            - candidates (np.ndarray): The actual selected candidate points.

    Notes:
        The function utilizes 'optimize_acqf_discrete_modified' for discrete optimization of the acquisition function.
        'NUM_RESTARTS' and 'RAW_SAMPLES' are set internally to control the optimization process.
        The function can handle different acquisition function strategies and can be adjusted for either maximizing or minimizing.
    """

    NUM_RESTARTS = 20
    RAW_SAMPLES = 512
    if acq_func == "GIBBON":
        acq_function = qLowerBoundMaxValueEntropy(
            model, X_candidate_BO, maximize=maximize
        )
    elif acq_func == "NEI":
        sampler = SobolQMCNormalSampler(1024)
        acq_function = qNoisyExpectedImprovement(model, torch.tensor(X_train), sampler)

    n_best = len(X_candidate_BO) // q

    candidates, acq_values = optimize_acqf_discrete_modified(
        acq_function=acq_function,
        bounds=bounds_norm,
        q=q,
        choices=X_candidate_BO,
        n_best=n_best,
        unique=True,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        sequential=sequential,
    )

    candidates = candidates.view(n_best, q, candidates.shape[2])
    acq_values = acq_values.view(n_best, q)

    index_set = []
    for return_nr in range(n_best):
        indices = find_indices(X_candidate_BO, candidates[return_nr])
        index_set.append(indices)

    index_set, acq_values, candidates = (
        np.array(index_set),
        np.array(acq_values),
        np.array(candidates),
    )

    return index_set, acq_values, candidates


def opt_acqfct_cost(
    X_train,
    model,
    X_candidate_BO,
    bounds_norm,
    q,
    LIGANDS_candidate_BO,
    price_dict_BO,
    acq_func="GIBBON",
):
    """
    Optimize the acquisition function considering the cost associated with each candidate set.

    This function first optimizes the acquisition function based on the given model and candidate points.
    It then rescales the acquisition values by the cost associated with each set of ligands, as defined by
    the `price_dict_BO`. The function returns the rearranged indices, acquisition values per price, and
    candidate sets, sorted in descending order of acquisition value per price.

    Parameters:
    - X_train (array): The training data points.
    - model: The Gaussian process model fitted to the training data.
    - X_candidate_BO (array): The candidate points for Bayesian optimization.
    - bounds_norm (array): Normalized bounds for the optimization problem.
    - q (int): The number of points to select in each iteration.
    - LIGANDS_candidate_BO (list): List of ligands corresponding to the candidate points.
    - price_dict_BO (dict): Dictionary mapping ligands to their respective costs.
    - acq_func (str, optional): The name of the acquisition function to use. Default is "GIBBON".

    Returns:
    - index_set_rearranged (array): The rearranged indices of the selected candidate points.
    - acq_values_per_price (array): The acquisition values divided by the price rescaling factor.
    - candidates_rearranged (array): The candidate points rearranged based on acquisition value per price.
    """

    index_set, acq_values, candidates = opt_acqfct(
        X_train,
        model,
        X_candidate_BO,
        bounds_norm,
        q,
        sequential=False,
        maximize=True,
        n_best=300,
        acq_func=acq_func,
    )

    ligand_set = []
    for subset in index_set:
        ligand_set.append(LIGANDS_candidate_BO[subset])

    price_rescaling_factors = []
    for ligands in ligand_set:
        price_rescaling_factors.append(function_cost(ligands, price_dict_BO))

    price_rescaling_factors = np.array(price_rescaling_factors)
    acq_values_per_price = acq_values / price_rescaling_factors
    row_sums_2 = acq_values_per_price.sum(axis=1)
    sorted_indices = np.argsort(row_sums_2)[::-1]

    index_set_rearranged = index_set[sorted_indices]
    acq_values_per_price = acq_values_per_price[sorted_indices]
    acq_values = acq_values[sorted_indices]
    candidates_rearranged = candidates[sorted_indices]

    return index_set_rearranged, acq_values_per_price, candidates_rearranged


def opt_acqfct_cost_B(
    X_train,
    model,
    X_candidate_BO,
    bounds_norm,
    q,
    LIGANDS_candidate_BO,
    ADDITIVES_candidate_BO,
    price_dict_BO_ligands,
    price_dict_BO_additives,
    acq_func="GIBBON",
):
    """
    Same as `opt_acqfct_cost`, but for the case of two cost functions i.e. ligands and additives.
    """

    index_set, acq_values, candidates = opt_acqfct(
        X_train,
        model,
        X_candidate_BO,
        bounds_norm,
        q,
        sequential=False,
        maximize=True,
        n_best=300,
        acq_func=acq_func,
    )

    ligand_set, additivies_set = [], []
    for subset in index_set:
        ligand_set.append(LIGANDS_candidate_BO[subset])
        additivies_set.append(ADDITIVES_candidate_BO[subset])

    price_rescaling_factors = []
    for ligands, additives in zip(ligand_set, additivies_set):
        cost_curr = function_cost_B(
            ligands, additives, price_dict_BO_ligands, price_dict_BO_additives
        )
        price_rescaling_factors.append(cost_curr)

    price_rescaling_factors = np.array(price_rescaling_factors)
    acq_values_per_price = acq_values / price_rescaling_factors
    row_sums_2 = acq_values_per_price.sum(axis=1)
    sorted_indices = np.argsort(row_sums_2)[::-1]

    index_set_rearranged = index_set[sorted_indices]
    acq_values_per_price = acq_values_per_price[sorted_indices]
    acq_values = acq_values[sorted_indices]
    candidates_rearranged = candidates[sorted_indices]

    return index_set_rearranged, acq_values_per_price, candidates_rearranged


def opt_gibbon(
    model,
    X_candidate_BO,
    bounds_norm,
    q,
    sequential=False,
    maximize=True,
    n_best=300,
    return_nr=0,
):
    """
    Perform optimization using the GIBBON acquisition function.

    This function optimizes the GIBBON acquisition function over a set of discrete candidate points.
    It uses a model to estimate the potential information gain in a Bayesian optimization context.
    The function returns the indices of the optimal points and the reshaped candidates based on the
    acquisition function values.

    Parameters:
    - model: The Gaussian process model fitted to the training data.
    - X_candidate_BO (array): The candidate points for Bayesian optimization.
    - bounds_norm (array): Normalized bounds for the optimization problem.
    - q (int): The number of points to select in each iteration.
    - sequential (bool, optional): Whether to select points sequentially or jointly. Default is False.
    - maximize (bool, optional): Whether to maximize or minimize the acquisition function. Default is True.
    - n_best (int, optional): The number of top candidate sets to consider. Default is 300.
    - return_nr (int, optional): The index of the candidate set to return. Default is 0.

    Returns:
    - indices (array): The indices of the optimal candidate points in the original candidate set.
    - candidates (array): The candidate points reshaped based on the acquisition values.
    """

    NUM_RESTARTS = 20
    RAW_SAMPLES = 512
    qGIBBON = qLowerBoundMaxValueEntropy(model, X_candidate_BO, maximize=maximize)
    n_best = len(X_candidate_BO) // q

    candidates, acq_values = optimize_acqf_discrete_modified(
        acq_function=qGIBBON,
        bounds=bounds_norm,
        q=q,
        choices=X_candidate_BO,
        n_best=n_best,
        unique=True,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        sequential=sequential,
    )

    candidates = candidates.view(n_best, q, candidates.shape[2])
    acq_values = acq_values.view(n_best, q)

    indices = find_indices(X_candidate_BO, candidates[return_nr])

    return indices, candidates


def gibbon_search(
    model, X_candidate_BO, bounds_norm, q, sequential=False, maximize=True
):
    """
    Conduct a search using the GIBBON acquisition function for Bayesian optimization.

    This function utilizes the GIBBON acquisition function to identify optimal candidate points
    from a given set. The function operates on a given Gaussian process model and a set of candidate points,
    returning the indices of the optimal candidates and the candidates themselves as determined by the
    acquisition function.

    Parameters:
    - model: The Gaussian process model used for predictions.
    - X_candidate_BO (array): Array of candidate points for Bayesian optimization.
    - bounds_norm (array): Normalized bounds of the search space.
    - q (int): The number of points to be selected in each iteration.
    - sequential (bool, optional): If True, points are selected sequentially; otherwise, jointly. Default is False.
    - maximize (bool, optional): Whether to maximize or minimize the acquisition function. Default is True.

    Returns:
    - indices (array): Indices of the optimal candidate points within the original candidate set.
    - candidates (array): The optimal candidate points as determined by the acquisition function.
    """

    NUM_RESTARTS = 20
    RAW_SAMPLES = 512
    qGIBBON = qLowerBoundMaxValueEntropy(model, X_candidate_BO, maximize=maximize)

    candidates, _ = optimize_acqf_discrete(
        acq_function=qGIBBON,
        bounds=bounds_norm,
        q=q,
        choices=X_candidate_BO,
        unique=True,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        sequential=sequential,
    )

    indices = find_indices(X_candidate_BO, candidates)

    return indices, candidates


def opt_qNEI(
    model, X_candidate_BO, bounds_norm, X_train, q, sequential=False, maximize=True
):
    """
    Optimize the Noisy Expected Improvement (NEI) acquisition function for a given model with discrete candidates.

    This function uses the Noisy Expected Improvement (qNEI) acquisition function to select optimal
    points in a Bayesian optimization context. It is particularly useful for batched Bayesian optimization
    where noisy evaluations of the objective function are expected.

    Parameters:
    - model: The Gaussian process model used for predictions.
    - X_candidate_BO (array): Array of candidate points for Bayesian optimization.
    - bounds_norm (array): Normalized bounds of the search space.
    - X_train (array): Training data points.
    - q (int): The number of points to be selected in each iteration (batch size).
    - sequential (bool, optional): If True, points are selected sequentially; otherwise, jointly. Default is False.
    - maximize (bool, optional): Whether to maximize or minimize the acquisition function. Default is True.

    Returns:
    - indices (array): Indices of the optimal candidate points within the original candidate set.
    - candidates (array): The optimal candidate points as determined by the acquisition function.

    The function utilizes a Sobol sequence-based QMC sampler for estimating the acquisition function and applies
    a discrete optimization technique to select the best candidates from the given set.
    """

    NUM_RESTARTS = 20
    RAW_SAMPLES = 512
    sampler = SobolQMCNormalSampler(1024)

    qLogNEI = qNoisyExpectedImprovement(model, torch.tensor(X_train), sampler)
    candidates, _ = optimize_acqf_discrete(
        acq_function=qLogNEI,
        bounds=bounds_norm,
        q=q,
        choices=X_candidate_BO,
        unique=True,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        sequential=sequential,
    )

    indices = find_indices(X_candidate_BO, candidates)

    return indices, candidates

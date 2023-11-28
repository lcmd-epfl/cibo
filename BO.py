import numpy as np
import torch
import warnings

from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf_discrete
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, LinearKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam

# Custom module imports
from botroch_ext import optimize_acqf_discrete_modified
from kernels import *
from utils import *

# Suppress warnings
warnings.filterwarnings("ignore")


random.seed(45577)
np.random.seed(4565777)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}


def find_indices(X_candidate_BO, candidates):
    """
    fuction finds indices of candidates in X_candidate_BO
    (as the acquisition function returns the candidates in a different order)
    Parameters:
        X_candidate_BO (numpy.ndarray): The holdout set.
        candidates (numpy.ndarray): The batch of molecules selected by the acquisition function.
    Returns:
        list: The indices of the selected molecules.
    """

    indices = []
    for candidate in candidates:
        indices.append(np.argwhere((X_candidate_BO == candidate).all(1)).flatten()[0])
    indices = np.array(indices)
    return indices


def update_model(
    X, y, bounds_norm, kernel_type="Tanimoto", fit_y=True, FIT_METHOD=True
):
    """
    Function that updates the GP model with new data with good presettings
    Parameters:
        X (numpy.ndarray): The training data.
        y (numpy.ndarray): The training labels.
        bounds_norm (numpy.ndarray): The bounds for normalization.
    Returns:
        model (botorch.models.gpytorch.GP): The updated GP model.
        scaler_y (TensorStandardScaler): The scaler for the labels.
    """
    GP_class = CustomGPModel(
        kernel_type=kernel_type,
        scale_type_X="botorch",
        bounds_norm=bounds_norm,
        fit_y=fit_y,
        FIT_METHOD=FIT_METHOD,
    )
    model = GP_class.fit(X, y)

    return model, GP_class.scaler_y


class TensorStandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance
    as defined in BoTorch
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


class CustomGPModel:
    def __init__(
        self,
        kernel_type="Matern",
        scale_type_X="sklearn",
        bounds_norm=None,
        fit_y=True,
        FIT_METHOD=True,
    ):
        self.kernel_type = kernel_type
        self.scale_type_X = scale_type_X
        self.bounds_norm = bounds_norm
        self.fit_y = fit_y

        self.FIT_METHOD = FIT_METHOD
        if not self.FIT_METHOD:
            if self.kernel_type == "RBF" or self.kernel_type == "Matern":
                self.NUM_EPOCHS_GD = 5000
            elif self.kernel_type == "Linear" or self.kernel_type == "Tanimoto":
                self.NUM_EPOCHS_GD = 1000

        if scale_type_X == "sklearn":
            self.scaler_X = MinMaxScaler()
        elif scale_type_X == "botorch":
            pass
        else:
            raise ValueError("Invalid scaler type")

        self.scaler_y = TensorStandardScaler()

    def fit(self, X_train, y_train):
        if self.scale_type_X == "sklearn":
            X_train = self.scaler_X.fit_transform(X_train)
        elif self.scale_type_X == "botorch":
            if type(X_train) == np.ndarray:
                X_train = torch.tensor(X_train, dtype=torch.float32)

        if self.fit_y:
            y_train = self.scaler_y.fit_transform(y_train)
        else:
            y_train = y_train

        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float64).view(-1, 1)

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

        # https://github.com/pytorch/botorch/blob/main/tutorials/fit_model_with_torch_optimizer.ipynb

        self.gp = InternalGP(self.X_train_tensor, self.y_train_tensor, kernel)
        if self.kernel_type == "Linear" or self.kernel_type == "Tanimoto":
            self.gp.likelihood.noise_constraint = gpytorch.constraints.GreaterThan(1e-3)

        if self.FIT_METHOD:
            """
            Use BoTorch fit method
            to fit the hyperparameters of the GP and the model weights
            """

            self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
            self.mll.to(self.X_train_tensor)
            fit_gpytorch_model(self.mll, max_retries=50000)

        else:
            """
            Use gradient descent to fit the hyperparameters of the GP with initial run
            to get reasonable hyperparameters and then a second run to get the best hyperparameters and model weights
            """

            self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
            self.mll.to(self.X_train_tensor)
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

                # If this loss is the best so far, update best_loss and best hyperparameters
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    if self.gp.covar_module.base_kernel.lengthscale != None:
                        best_lengthscale = lengthscale
                    best_noise = noise
                # print(f"Finished grid point with lengthscale {lengthscale}, noise {noise}, loss {loss.item()}")

            # Set the best found hyperparameters
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
                # calculate the negative log likelihood
                loss = -self.mll(output, self.y_train_tensor.flatten())
                # back prop gradients
                loss.backward()
                # print every 10 iterations
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


def gibbon_search_modified_all(
    model, X_candidate_BO, bounds_norm, q, sequential=False, maximize=True, n_best=300
):
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

    index_set = []
    for return_nr in range(n_best):
        indices = find_indices(X_candidate_BO, candidates[return_nr])
        index_set.append(indices)

    index_set = np.array(index_set)

    return index_set, candidates


def gibbon_search_modified(
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
    https://botorch.org/tutorials/GIBBON_for_efficient_batch_entropy_search
    returns index of the q best candidates in X_candidate_BO
    as well as their feature vectors using a modified version of the GIBBON function
    implemented in BoTorch: it returns the n_best best candidates and then selects the
    return_nr-th best candidate as the batch of molecules to be selected instead of only the
    best candidate
    source here https://botorch.org/api/_modules/botorch/optim/optimize.html#optimize_acqf_discrete
    but using here optimize_acqf_discrete_modified

    Parameters:
       model (botorch.models.gpytorch.GP): The GP model.
       X_candidate_BO (numpy.ndarray): The holdout set.
       bounds_norm (numpy.ndarray): The bounds for normalization.
       q (int): The batch size.
       sequential (bool): Whether to use sequential optimization.
       maximize (bool): Whether to maximize or minimize the acquisition function.
    Returns:
       nump.ndarray: The indices of the selected molecules.
       nump.ndarray: The selected molecules.
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
    #https://botorch.org/tutorials/GIBBON_for_efficient_batch_entropy_search
    returns index of the q best candidates in X_candidate_BO
    as well as their feature vectors
    source here https://botorch.org/api/_modules/botorch/optim/optimize.html#optimize_acqf_discrete
    Parameters:
        model (botorch.models.gpytorch.GP): The GP model.
        X_candidate_BO (numpy.ndarray): The holdout set.
        bounds_norm (numpy.ndarray): The bounds for normalization.
        q (int): The batch size.
        sequential (bool): Whether to use sequential optimization.
        maximize (bool): Whether to maximize or minimize the acquisition function.
    Returns:
        nump.ndarray: The indices of the selected molecules.
        nump.ndarray: The selected molecules.
    """

    NUM_RESTARTS = 20
    RAW_SAMPLES = 512
    qGIBBON = qLowerBoundMaxValueEntropy(model, X_candidate_BO, maximize=maximize)

    candidates, best_acq_values = optimize_acqf_discrete(
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


def qNoisyEI_search(
    model, X_candidate_BO, bounds_norm, X_train, q, sequential=False, maximize=True
):
    """
    Noisy expected improvement with batches
    """

    NUM_RESTARTS = 20
    RAW_SAMPLES = 512
    sampler = SobolQMCNormalSampler(1024)

    qLogNEI = qNoisyExpectedImprovement(model, torch.tensor(X_train), sampler)  # , q=q)
    candidates, best_acq_values = optimize_acqf_discrete(
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

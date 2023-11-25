import numpy as np
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, LinearKernel
from gpytorch.means import ConstantMean
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam
from itertools import combinations
from botorch.optim import optimize_acqf_discrete, optimize_acqf_discrete_modified
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from utils import *
from kernels import *
import warnings

warnings.filterwarnings("ignore")

random.seed(45577)
np.random.seed(4565777)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}


def select_batch(suggested_costs, MAX_BATCH_COST, BATCH_SIZE):
    """
    Selects a batch of molecules from a list of suggested molecules that have the lowest indices
    while meeting the constraints of the maximum cost and batch size.
    """

    n = len(suggested_costs)
    # Check if BATCH_SIZE is larger than the length of the array, if so return None
    if BATCH_SIZE > n:
        return []
    valid_combinations = []
    # Find all valid combinations that meet the cost condition
    for indices in combinations(range(n), BATCH_SIZE):
        if sum(suggested_costs[i] for i in indices) <= MAX_BATCH_COST:
            valid_combinations.append(indices)
    # If there are no valid combinations, return None
    if not valid_combinations:
        return []
    # Select the combination with the lowest indices
    best_indices = min(
        valid_combinations, key=lambda x: tuple(suggested_costs[i] for i in x)
    )
    return list(best_indices)


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


def RS_STEP(RANDOM_data):
    # Extract the data from the dictionary
    y_candidate_RANDOM = RANDOM_data["y_candidate_RANDOM"]
    y_best_RANDOM = RANDOM_data["y_best_RANDOM"]
    costs_RANDOM = RANDOM_data["costs_RANDOM"]
    BATCH_SIZE = RANDOM_data["BATCH_SIZE"]
    y_better_RANDOM = RANDOM_data["y_better_RANDOM"]
    running_costs_RANDOM = RANDOM_data["running_costs_RANDOM"]

    indices_random = np.random.choice(
        np.arange(len(y_candidate_RANDOM)), size=BATCH_SIZE, replace=False
    )
    if max(y_candidate_RANDOM[indices_random])[0] > y_best_RANDOM:
        y_best_RANDOM = max(y_candidate_RANDOM[indices_random])[0]

    y_better_RANDOM.append(y_best_RANDOM)
    BATCH_COST = sum(costs_RANDOM[indices_random])[0]
    running_costs_RANDOM.append(running_costs_RANDOM[-1] + BATCH_COST)
    y_candidate_RANDOM = np.delete(y_candidate_RANDOM, indices_random, axis=0)
    costs_RANDOM = np.delete(costs_RANDOM, indices_random, axis=0)

    # Update the dictionary with the new values

    RANDOM_data["y_candidate_RANDOM"] = y_candidate_RANDOM
    RANDOM_data["y_better_RANDOM"] = y_better_RANDOM
    RANDOM_data["y_best_RANDOM"] = y_best_RANDOM
    RANDOM_data["running_costs_RANDOM"] = running_costs_RANDOM
    RANDOM_data["costs_RANDOM"] = costs_RANDOM

    # There is no need to update BATCH_SIZE and MAX_BATCH_COST as they are constants and do not change

    # Return the updated dictionary
    return RANDOM_data


def BO_CASE_1_STEP(BO_data):
    """
    Simple BO step without any cost constraints but keep track of the costs
    """
    # Get current BO data from last iteration
    model = BO_data["model"]
    X, y = BO_data["X"], BO_data["y"]
    N_train = BO_data["N_train"]
    y_candidate_BO = BO_data["y_candidate_BO"]
    X_candidate_BO = BO_data["X_candidate_BO"]
    bounds_norm = BO_data["bounds_norm"]
    BATCH_SIZE = BO_data["BATCH_SIZE"]
    y_best_BO = BO_data["y_best_BO"]
    y_better_BO = BO_data["y_better_BO"]
    costs_BO = BO_data["costs_BO"]
    running_costs_BO = BO_data["running_costs_BO"]
    scaler_y = BO_data["scaler_y"]

    # Get new candidates
    indices, candidates = gibbon_search(
        model, X_candidate_BO, bounds_norm, q=BATCH_SIZE
    )

    X, y = update_X_y(X, y, candidates, y_candidate_BO, indices)
    y_best_BO = check_better(y, y_best_BO)
    y_better_BO.append(y_best_BO)
    running_costs_BO.append((running_costs_BO[-1] + sum(costs_BO[indices]))[0])
    # Update model
    model, scaler_y = update_model(X, y, bounds_norm)
    # Delete candidates from pool of candidates since added to training data
    X_candidate_BO = np.delete(X_candidate_BO, indices, axis=0)
    y_candidate_BO = np.delete(y_candidate_BO, indices, axis=0)
    costs_BO = np.delete(costs_BO, indices, axis=0)

    # Update BO data for next iteration
    BO_data["model"] = model
    BO_data["X"], BO_data["y"] = X, y
    BO_data["N_train"] = len(X)
    BO_data["y_candidate_BO"] = y_candidate_BO
    BO_data["X_candidate_BO"] = X_candidate_BO
    BO_data["bounds_norm"] = bounds_norm
    BO_data["y_best_BO"] = y_best_BO
    BO_data["y_better_BO"] = y_better_BO
    BO_data["costs_BO"] = costs_BO
    BO_data["running_costs_BO"] = running_costs_BO
    BO_data["scaler_y"] = scaler_y

    return BO_data


def BO_AWARE_SCAN_FAST_CASE_1_STEP(BO_data):
    """
    BO with cost constraints on the costs per batch
    """
    model = BO_data["model"]
    X, y = BO_data["X"], BO_data["y"]
    N_train = BO_data["N_train"]
    y_candidate_BO = BO_data["y_candidate_BO"]
    X_candidate_BO = BO_data["X_candidate_BO"]
    bounds_norm = BO_data["bounds_norm"]
    BATCH_SIZE = BO_data["BATCH_SIZE"]
    y_best_BO = BO_data["y_best_BO"]
    y_better_BO = BO_data["y_better_BO"]
    costs_BO = BO_data["costs_BO"]
    running_costs_BO = BO_data["running_costs_BO"]
    scaler_y = BO_data["scaler_y"]
    MAX_BATCH_COST = BO_data["MAX_BATCH_COST"]

    index_set, _ = gibbon_search_modified_all(
        model,
        X_candidate_BO,
        bounds_norm,
        q=BATCH_SIZE,
        sequential=False,
        maximize=True,
    )

    for indices in index_set:
        suggested_costs = costs_BO[indices].flatten()

        if suggested_costs.sum() <= MAX_BATCH_COST:
            X, y = update_X_y(
                X,
                y,
                X_candidate_BO[indices],
                y_candidate_BO,
                indices,
            )
            y_best_BO = check_better(y, y_best_BO)

            y_better_BO.append(y_best_BO)
            BATCH_COST = sum(costs_BO[indices])[0]
            print("Batch cost1: ", BATCH_COST)
            running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
            model, scaler_y = update_model(X, y, bounds_norm)

            X_candidate_BO = np.delete(X_candidate_BO, indices, axis=0)
            y_candidate_BO = np.delete(y_candidate_BO, indices, axis=0)
            costs_BO = np.delete(costs_BO, indices, axis=0)
            break

    # Update BO data for next iteration
    BO_data["model"] = model
    BO_data["X"], BO_data["y"] = X, y
    BO_data["N_train"] = len(X)
    BO_data["y_candidate_BO"] = y_candidate_BO
    BO_data["X_candidate_BO"] = X_candidate_BO
    BO_data["bounds_norm"] = bounds_norm
    BO_data["y_best_BO"] = y_best_BO
    BO_data["y_better_BO"] = y_better_BO
    BO_data["costs_BO"] = costs_BO
    BO_data["running_costs_BO"] = running_costs_BO
    BO_data["scaler_y"] = scaler_y

    return BO_data


def BO_AWARE_SCAN_CASE_1_STEP(BO_data):
    """
    BO with cost constraints on the costs per batch
    """
    model = BO_data["model"]
    X, y = BO_data["X"], BO_data["y"]
    N_train = BO_data["N_train"]
    y_candidate_BO = BO_data["y_candidate_BO"]
    X_candidate_BO = BO_data["X_candidate_BO"]
    bounds_norm = BO_data["bounds_norm"]
    BATCH_SIZE = BO_data["BATCH_SIZE"]
    y_best_BO = BO_data["y_best_BO"]
    y_better_BO = BO_data["y_better_BO"]
    costs_BO = BO_data["costs_BO"]
    running_costs_BO = BO_data["running_costs_BO"]
    scaler_y = BO_data["scaler_y"]
    MAX_BATCH_COST = BO_data["MAX_BATCH_COST"]

    SUCCESS = False
    ITERATION = 0

    while True:
        print(ITERATION)
        SUCCESS_1 = False
        indices, candidates = gibbon_search_modified(
            model,
            X_candidate_BO,
            bounds_norm,
            q=BATCH_SIZE,
            sequential=False,
            maximize=True,
            n_best=300,
            return_nr=ITERATION,
        )
        suggested_costs = costs_BO[indices].flatten()

        if suggested_costs.sum() <= MAX_BATCH_COST:
            X, y = update_X_y(
                X,
                y,
                X_candidate_BO[indices],
                y_candidate_BO,
                indices,
            )
            y_best_BO = check_better(y, y_best_BO)

            y_better_BO.append(y_best_BO)
            BATCH_COST = sum(costs_BO[indices])[0]
            print("Batch cost1: ", BATCH_COST)
            running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
            model, scaler_y = update_model(X, y, bounds_norm)

            X_candidate_BO = np.delete(X_candidate_BO, indices, axis=0)
            y_candidate_BO = np.delete(y_candidate_BO, indices, axis=0)
            costs_BO = np.delete(costs_BO, indices, axis=0)
            break

        ITERATION += 1

    # Update BO data for next iteration
    BO_data["model"] = model
    BO_data["X"], BO_data["y"] = X, y
    BO_data["N_train"] = len(X)
    BO_data["y_candidate_BO"] = y_candidate_BO
    BO_data["X_candidate_BO"] = X_candidate_BO
    BO_data["bounds_norm"] = bounds_norm
    BO_data["y_best_BO"] = y_best_BO
    BO_data["y_better_BO"] = y_better_BO
    BO_data["costs_BO"] = costs_BO
    BO_data["running_costs_BO"] = running_costs_BO
    BO_data["scaler_y"] = scaler_y

    return BO_data


def BO_AWARE_CASE_1_STEP(BO_data):
    """
    BO with cost constraints on the costs per batch
    """
    model = BO_data["model"]
    X, y = BO_data["X"], BO_data["y"]
    N_train = BO_data["N_train"]
    y_candidate_BO = BO_data["y_candidate_BO"]
    X_candidate_BO = BO_data["X_candidate_BO"]
    bounds_norm = BO_data["bounds_norm"]
    BATCH_SIZE = BO_data["BATCH_SIZE"]
    y_best_BO = BO_data["y_best_BO"]
    y_better_BO = BO_data["y_better_BO"]
    costs_BO = BO_data["costs_BO"]
    running_costs_BO = BO_data["running_costs_BO"]
    scaler_y = BO_data["scaler_y"]
    MAX_BATCH_COST = BO_data["MAX_BATCH_COST"]

    SUCCESS = False
    indices, candidates = gibbon_search(
        model, X_candidate_BO, bounds_norm, q=BATCH_SIZE
    )
    suggested_costs = costs_BO[indices].flatten()
    cheap_indices = select_batch(suggested_costs, MAX_BATCH_COST, BATCH_SIZE)
    cheap_indices, SUCCESS_1 = check_success(cheap_indices, indices)
    ITERATION = 1

    while (cheap_indices == []) or (len(cheap_indices) < BATCH_SIZE):
        INCREMENTED_MAX_BATCH_COST = MAX_BATCH_COST
        SUCCESS_1 = False

        INCREMENTED_BATCH_SIZE = BATCH_SIZE + ITERATION
        print("Incrementing canditates for batch to: ", INCREMENTED_BATCH_SIZE)
        if INCREMENTED_BATCH_SIZE > len(X_candidate_BO):
            print("Not enough candidates left to account for the costs")
            # therefore increasing the max batch cost to finally get enough candidates
            INCREMENTED_MAX_BATCH_COST += 1

        indices, candidates = gibbon_search(
            model, X_candidate_BO, bounds_norm, q=INCREMENTED_BATCH_SIZE
        )
        suggested_costs = costs_BO[indices].flatten()

        cheap_indices = select_batch(
            suggested_costs, INCREMENTED_MAX_BATCH_COST, BATCH_SIZE
        )
        cheap_indices, SUCCESS_2 = check_success(cheap_indices, indices)

        if (cheap_indices != []) and len(cheap_indices) == BATCH_SIZE:
            X, y = update_X_y(
                X,
                y,
                X_candidate_BO[cheap_indices],
                y_candidate_BO,
                cheap_indices,
            )
            y_best_BO = check_better(y, y_best_BO)

            y_better_BO.append(y_best_BO)
            BATCH_COST = sum(costs_BO[cheap_indices])[0]
            print("Batch cost1: ", BATCH_COST)
            running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
            model, scaler_y = update_model(X, y, bounds_norm)

            X_candidate_BO = np.delete(X_candidate_BO, cheap_indices, axis=0)
            y_candidate_BO = np.delete(y_candidate_BO, cheap_indices, axis=0)
            costs_BO = np.delete(costs_BO, cheap_indices, axis=0)
            break

        ITERATION += 1

    if SUCCESS_1:
        X, y = update_X_y(
            X,
            y,
            X_candidate_BO[cheap_indices],
            y_candidate_BO,
            cheap_indices,
        )
        y_best_BO = check_better(y, y_best_BO)
        y_better_BO.append(y_best_BO)
        BATCH_COST = sum(costs_BO[cheap_indices])[0]
        print("Batch cost2: ", BATCH_COST)
        running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
        model, scaler_y = update_model(X, y, bounds_norm)
        X_candidate_BO = np.delete(X_candidate_BO, cheap_indices, axis=0)
        y_candidate_BO = np.delete(y_candidate_BO, cheap_indices, axis=0)
        costs_BO = np.delete(costs_BO, cheap_indices, axis=0)

    # Update BO data for next iteration
    BO_data["model"] = model
    BO_data["X"], BO_data["y"] = X, y
    BO_data["N_train"] = len(X)
    BO_data["y_candidate_BO"] = y_candidate_BO
    BO_data["X_candidate_BO"] = X_candidate_BO
    BO_data["bounds_norm"] = bounds_norm
    BO_data["y_best_BO"] = y_best_BO
    BO_data["y_better_BO"] = y_better_BO
    BO_data["costs_BO"] = costs_BO
    BO_data["running_costs_BO"] = running_costs_BO
    BO_data["scaler_y"] = scaler_y

    return BO_data


def BO_CASE_2A_STEP(BO_data):
    """
    Normal BO with no cost constraints but keep track of the costs per batch for ca
    """
    # Get current BO data from last iteration
    model = BO_data["model"]
    X, y = BO_data["X"], BO_data["y"]
    N_train = BO_data["N_train"]
    y_candidate_BO = BO_data["y_candidate_BO"]
    X_candidate_BO = BO_data["X_candidate_BO"]
    bounds_norm = BO_data["bounds_norm"]
    BATCH_SIZE = BO_data["BATCH_SIZE"]
    y_best_BO = BO_data["y_best_BO"]
    y_better_BO = BO_data["y_better_BO"]
    LIGANDS_candidate_BO = BO_data["LIGANDS_candidate_BO"]
    price_dict_BO = BO_data["price_dict_BO"]
    running_costs_BO = BO_data["running_costs_BO"]
    scaler_y = BO_data["scaler_y"]

    # Assuming gibbon_search, update_X_y, compute_price_acquisition_ligands, check_better, update_model, and update_price_dict_ligands are defined elsewhere
    indices, candidates = gibbon_search(
        model, X_candidate_BO, bounds_norm, q=BATCH_SIZE
    )
    X, y = update_X_y(X, y, candidates, y_candidate_BO, indices)
    NEW_LIGANDS = LIGANDS_candidate_BO[indices]
    suggested_costs_all, _ = compute_price_acquisition_ligands(
        NEW_LIGANDS, price_dict_BO
    )
    y_best_BO = check_better(y, y_best_BO)
    y_better_BO.append(y_best_BO)
    running_costs_BO.append((running_costs_BO[-1] + suggested_costs_all))
    model, scaler_y = update_model(X, y, bounds_norm)
    X_candidate_BO = np.delete(X_candidate_BO, indices, axis=0)
    y_candidate_BO = np.delete(y_candidate_BO, indices, axis=0)
    LIGANDS_candidate_BO = np.delete(LIGANDS_candidate_BO, indices, axis=0)
    price_dict_BO = update_price_dict_ligands(price_dict_BO, NEW_LIGANDS)

    # Update BO data for next iteration
    BO_data["model"] = model
    BO_data["X"] = X
    BO_data["y"] = y
    BO_data["y_candidate_BO"] = y_candidate_BO
    BO_data["X_candidate_BO"] = X_candidate_BO
    BO_data["y_best_BO"] = y_best_BO
    BO_data["y_better_BO"] = y_better_BO
    BO_data["LIGANDS_candidate_BO"] = LIGANDS_candidate_BO
    BO_data["price_dict_BO"] = price_dict_BO
    BO_data["running_costs_BO"] = running_costs_BO
    BO_data["N_train"] = len(X)
    BO_data["scaler_y"] = scaler_y

    return BO_data


def RS_STEP_2A(RANDOM_data):
    y_candidate_RANDOM = RANDOM_data["y_candidate_RANDOM"]
    BATCH_SIZE = RANDOM_data["BATCH_SIZE"]
    LIGANDS_candidate_RANDOM = RANDOM_data["LIGANDS_candidate_RANDOM"]
    price_dict_RANDOM = RANDOM_data["price_dict_RANDOM"]
    y_best_RANDOM = RANDOM_data["y_best_RANDOM"]
    y_better_RANDOM = RANDOM_data["y_better_RANDOM"]
    running_costs_RANDOM = RANDOM_data["running_costs_RANDOM"]

    indices_random = np.random.choice(
        np.arange(len(y_candidate_RANDOM)), size=BATCH_SIZE, replace=False
    )
    NEW_LIGANDS = LIGANDS_candidate_RANDOM[indices_random]
    suggested_costs_all, price_per_ligand = compute_price_acquisition_ligands(
        NEW_LIGANDS, price_dict_RANDOM
    )

    if max(y_candidate_RANDOM[indices_random])[0] > y_best_RANDOM:
        y_best_RANDOM = max(y_candidate_RANDOM[indices_random])[0]

    y_better_RANDOM.append(y_best_RANDOM)
    running_costs_RANDOM.append(running_costs_RANDOM[-1] + suggested_costs_all)

    y_candidate_RANDOM = np.delete(y_candidate_RANDOM, indices_random, axis=0)
    LIGANDS_candidate_RANDOM = np.delete(
        LIGANDS_candidate_RANDOM, indices_random, axis=0
    )
    price_dict_RANDOM = update_price_dict_ligands(price_dict_RANDOM, NEW_LIGANDS)

    # Update all modified quantities and return RANDOM_data
    RANDOM_data["y_candidate_RANDOM"] = y_candidate_RANDOM
    RANDOM_data["LIGANDS_candidate_RANDOM"] = LIGANDS_candidate_RANDOM
    RANDOM_data["price_dict_RANDOM"] = price_dict_RANDOM
    RANDOM_data["y_best_RANDOM"] = y_best_RANDOM
    RANDOM_data["y_better_RANDOM"] = y_better_RANDOM
    RANDOM_data["running_costs_RANDOM"] = running_costs_RANDOM

    return RANDOM_data


def BO_AWARE_CASE_2A_STEP(BO_data):
    """
    BO with cost constraints on the costs per batch, updating prices when ligand first bought
    """

    # Get current BO data from last iteration
    model = BO_data["model"]
    X, y = BO_data["X"], BO_data["y"]
    N_train = BO_data["N_train"]
    y_candidate_BO = BO_data["y_candidate_BO"]
    X_candidate_BO = BO_data["X_candidate_BO"]
    bounds_norm = BO_data["bounds_norm"]
    BATCH_SIZE = BO_data["BATCH_SIZE"]
    y_best_BO = BO_data["y_best_BO"]
    y_better_BO = BO_data["y_better_BO"]
    LIGANDS_candidate_BO = BO_data["LIGANDS_candidate_BO"]
    price_dict_BO = BO_data["price_dict_BO"]
    running_costs_BO = BO_data["running_costs_BO"]
    MAX_BATCH_COST = BO_data["MAX_BATCH_COST"]
    scaler_y = BO_data["scaler_y"]

    SUCCESS_1 = False
    indices, candidates = gibbon_search(
        model, X_candidate_BO, bounds_norm, q=BATCH_SIZE
    )
    NEW_LIGANDS = LIGANDS_candidate_BO[indices]
    (
        suggested_costs_all,
        price_per_ligand,
    ) = compute_price_acquisition_ligands(NEW_LIGANDS, price_dict_BO)
    cheap_indices_1 = select_batch(price_per_ligand, MAX_BATCH_COST, BATCH_SIZE)
    cheap_indices, SUCCESS_1 = check_success(cheap_indices_1, indices)

    if SUCCESS_1:
        BATCH_COST = np.array(price_per_ligand)[cheap_indices_1].sum()

    ITERATION = 1

    while (cheap_indices == []) or (len(cheap_indices) < BATCH_SIZE):
        INCREMENTED_MAX_BATCH_COST = MAX_BATCH_COST
        SUCCESS_1 = False

        INCREMENTED_BATCH_SIZE = BATCH_SIZE + ITERATION
        print(
            "Incrementing canditates for batch to: ",
            INCREMENTED_BATCH_SIZE,
        )
        if INCREMENTED_BATCH_SIZE > len(X_candidate_BO):
            print("Not enough candidates left to account for the costs")
            INCREMENTED_MAX_BATCH_COST += 1
        if INCREMENTED_BATCH_SIZE > 50:
            print(
                "After 50 iterations, still cost conditions not met. Increasing cost by 1 and trying again"
            )
            INCREMENTED_MAX_BATCH_COST += 1

        indices, candidates = gibbon_search(
            model, X_candidate_BO, bounds_norm, q=INCREMENTED_BATCH_SIZE
        )
        NEW_LIGANDS = LIGANDS_candidate_BO[indices]
        (
            suggested_costs_all,
            price_per_ligand,
        ) = compute_price_acquisition_ligands(NEW_LIGANDS, price_dict_BO)

        cheap_indices_1 = select_batch(
            price_per_ligand, INCREMENTED_MAX_BATCH_COST, BATCH_SIZE
        )
        cheap_indices, SUCCESS_2 = check_success(cheap_indices_1, indices)
        BATCH_COST = np.array(price_per_ligand)[cheap_indices_1].sum()

        if (cheap_indices != []) and len(cheap_indices) == BATCH_SIZE:
            X, y = update_X_y(
                X,
                y,
                X_candidate_BO[cheap_indices],
                y_candidate_BO,
                cheap_indices,
            )
            y_best_BO = check_better(y, y_best_BO)
            y_better_BO.append(y_best_BO)

            print("Batch cost1: ", BATCH_COST)

            running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
            model, scaler_y = update_model(X, y, bounds_norm)
            X_candidate_BO = np.delete(X_candidate_BO, cheap_indices, axis=0)
            y_candidate_BO = np.delete(y_candidate_BO, cheap_indices, axis=0)
            LIGANDS_candidate_BO = np.delete(
                LIGANDS_candidate_BO, cheap_indices, axis=0
            )
            price_dict_BO = update_price_dict_ligands(
                price_dict_BO, NEW_LIGANDS[cheap_indices_1]
            )

        ITERATION += 1

    if SUCCESS_1:
        X, y = update_X_y(
            X,
            y,
            X_candidate_BO[cheap_indices],
            y_candidate_BO,
            cheap_indices,
        )
        y_best_BO = check_better(y, y_best_BO)
        y_better_BO.append(y_best_BO)

        print("Batch cost2: ", BATCH_COST)
        running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
        model, scaler_y = update_model(X, y, bounds_norm)
        X_candidate_BO = np.delete(X_candidate_BO, cheap_indices, axis=0)
        y_candidate_BO = np.delete(y_candidate_BO, cheap_indices, axis=0)
        LIGANDS_candidate_BO = np.delete(LIGANDS_candidate_BO, cheap_indices, axis=0)
        price_dict_BO = update_price_dict_ligands(
            price_dict_BO, NEW_LIGANDS[cheap_indices_1]
        )

    # Update BO data for next iteration
    BO_data["model"] = model
    BO_data["X"] = X
    BO_data["y"] = y
    BO_data["y_candidate_BO"] = y_candidate_BO
    BO_data["X_candidate_BO"] = X_candidate_BO
    BO_data["y_best_BO"] = y_best_BO
    BO_data["y_better_BO"] = y_better_BO
    BO_data["LIGANDS_candidate_BO"] = LIGANDS_candidate_BO
    BO_data["price_dict_BO"] = price_dict_BO
    BO_data["running_costs_BO"] = running_costs_BO
    BO_data["N_train"] = len(X)
    BO_data["scaler_y"] = scaler_y

    return BO_data


def BO_AWARE_SCAN_FAST_CASE_2_STEP(BO_data):
    # Get current BO data from last iteration
    model = BO_data["model"]
    X, y = BO_data["X"], BO_data["y"]
    N_train = BO_data["N_train"]
    y_candidate_BO = BO_data["y_candidate_BO"]
    X_candidate_BO = BO_data["X_candidate_BO"]
    bounds_norm = BO_data["bounds_norm"]
    BATCH_SIZE = BO_data["BATCH_SIZE"]
    y_best_BO = BO_data["y_best_BO"]
    y_better_BO = BO_data["y_better_BO"]
    LIGANDS_candidate_BO = BO_data["LIGANDS_candidate_BO"]
    price_dict_BO = BO_data["price_dict_BO"]
    running_costs_BO = BO_data["running_costs_BO"]
    MAX_BATCH_COST = BO_data["MAX_BATCH_COST"]
    scaler_y = BO_data["scaler_y"]

    index_set, _ = gibbon_search_modified_all(
        model,
        X_candidate_BO,
        bounds_norm,
        q=BATCH_SIZE,
        sequential=False,
        maximize=True,
    )

    price_list = np.array(list(price_dict_BO.values()))
    non_zero_prices = price_list[price_list > 0]
    if len(non_zero_prices) > 0:
        index_of_smallest_nonzero = np.where(price_list == non_zero_prices.min())[0][0]
        cheapest_ligand_price = price_list[index_of_smallest_nonzero]
        cheapest_ligand = list(price_dict_BO.keys())[index_of_smallest_nonzero]
        sorted_non_zero_prices = np.sort(non_zero_prices)
        if not len(sorted_non_zero_prices) > 1:
            print("Only one ligand left")

        if cheapest_ligand_price > MAX_BATCH_COST:
            print("No ligand can be bought with the current budget")
            print("Ask your boss for more $$$")

    else:
        print("All ligands have been bought")

    # select cheapest one that is not already 0 (correct that in the initialization)
    SUCCESS = False
    for indices in index_set:
        NEW_LIGANDS = LIGANDS_candidate_BO[indices]
        suggested_costs_all, price_per_ligand = compute_price_acquisition_ligands(
            NEW_LIGANDS, price_dict_BO
        )

        BATCH_COST = suggested_costs_all
        if suggested_costs_all <= MAX_BATCH_COST:
            X, y = update_X_y(
                X,
                y,
                X_candidate_BO[indices],
                y_candidate_BO,
                indices,
            )
            y_best_BO = check_better(y, y_best_BO)
            y_better_BO.append(y_best_BO)

            print("Batch cost1: ", BATCH_COST)

            running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
            model, scaler_y = update_model(X, y, bounds_norm)
            X_candidate_BO = np.delete(X_candidate_BO, indices, axis=0)
            y_candidate_BO = np.delete(y_candidate_BO, indices, axis=0)
            LIGANDS_candidate_BO = np.delete(LIGANDS_candidate_BO, indices, axis=0)
            price_dict_BO = update_price_dict_ligands(price_dict_BO, NEW_LIGANDS)
            SUCCESS = True
            break

    if not SUCCESS:
        # means that only mixed ligand batches are suggested by the acqfct which we cannot afford, 
        # thus take points from the cheapest ligand
        if cheapest_ligand_price < MAX_BATCH_COST:
            # find indices where LIGANDS_candidate_BO == cheapest_ligand
            NEW_LIGANDS = [cheapest_ligand]
            indices_cheap = np.where(LIGANDS_candidate_BO == cheapest_ligand)[0]

            index, _ = gibbon_search(
                model, X_candidate_BO[indices_cheap], bounds_norm, q=5)

            X, y = update_X_y(
                X,
                y,
                X_candidate_BO[indices_cheap][index],
                y_candidate_BO,
                indices_cheap[index],
            )
            y_best_BO = check_better(y, y_best_BO)
            y_better_BO.append(y_best_BO)

            BATCH_COST = cheapest_ligand_price
            print("Batch cost2: ", BATCH_COST)
            running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
            model, scaler_y = update_model(X, y, bounds_norm)
            X_candidate_BO = np.delete(X_candidate_BO, indices_cheap[index], axis=0)
            y_candidate_BO = np.delete(y_candidate_BO, indices_cheap[index], axis=0)
            LIGANDS_candidate_BO = np.delete(
                LIGANDS_candidate_BO, indices_cheap[index], axis=0
            )
            price_dict_BO = update_price_dict_ligands(price_dict_BO, NEW_LIGANDS)
        else:
            # TODO implement case where you just do free measurements with the ligands already bought
            index_of_zero = np.where(price_list == 0)[0][0]
            cheapest_ligand = list(price_dict_BO.keys())[index_of_zero]
            indices_cheap = np.where(LIGANDS_candidate_BO == cheapest_ligand)[0]
            if indices_cheap > 0:
                index, _ = gibbon_search(model, X_candidate_BO[indices_cheap], bounds_norm, q=5)
                X, y = update_X_y(
                    X,
                    y,
                    X_candidate_BO[indices_cheap][index],
                    y_candidate_BO,
                    indices_cheap[index],
                )
                y_best_BO = check_better(y, y_best_BO)
                y_better_BO.append(y_best_BO)

                BATCH_COST = 0
                print("Batch cost3: ", BATCH_COST)
                running_costs_BO.append(running_costs_BO[-1] + BATCH_COST)
                model, scaler_y = update_model(X, y, bounds_norm)
                X_candidate_BO = np.delete(X_candidate_BO, indices_cheap[index], axis=0)
                y_candidate_BO = np.delete(y_candidate_BO, indices_cheap[index], axis=0)
                LIGANDS_candidate_BO = np.delete(
                    LIGANDS_candidate_BO, indices_cheap[index], axis=0
                )
                price_dict_BO = update_price_dict_ligands(price_dict_BO, NEW_LIGANDS)
            else: 
                print("All affordable ligands have been bought and no more free measurements possible. BO will stagnate now.")
    # Update BO data for next iteration
    BO_data["model"] = model
    BO_data["X"] = X
    BO_data["y"] = y
    BO_data["y_candidate_BO"] = y_candidate_BO
    BO_data["X_candidate_BO"] = X_candidate_BO
    BO_data["y_best_BO"] = y_best_BO
    BO_data["y_better_BO"] = y_better_BO
    BO_data["LIGANDS_candidate_BO"] = LIGANDS_candidate_BO
    BO_data["price_dict_BO"] = price_dict_BO
    BO_data["running_costs_BO"] = running_costs_BO
    BO_data["N_train"] = len(X)
    BO_data["scaler_y"] = scaler_y

    return BO_data
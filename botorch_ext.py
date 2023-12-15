import torch
import numpy as np
from typing import Any, Tuple, Optional, Callable, List
from torch import Tensor
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.gpytorch import Model
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
import xgboost as xgb


def optimize_acqf_discrete_modified(
    acq_function: AcquisitionFunction,
    q: int,
    choices: Tensor,
    n_best: int,  # Specify how many best results to return
    max_batch_size: int = 2048,
    unique: bool = True,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    # [Existing documentation and initial checks]

    choices_batched = choices.unsqueeze(-2)

    if q > 1:
        candidate_list, acq_value_list = [], []
        base_X_pending = acq_function.X_pending
        for _ in range(q):
            with torch.no_grad():
                acq_values = _split_batch_eval_acqf(
                    acq_function=acq_function,
                    X=choices_batched,
                    max_batch_size=max_batch_size,
                )
            # Sort acq_values and get indices of the top n best values
            sorted_indices = torch.argsort(acq_values, descending=True)[:n_best]
            best_candidates = choices_batched[sorted_indices]
            best_acq_values = acq_values[sorted_indices]

            candidate_list.append(best_candidates)
            acq_value_list.append(best_acq_values)

            # Enforce uniqueness by removing the selected choices
            if unique:
                mask = torch.ones(choices_batched.shape[0], dtype=torch.bool)
                mask[sorted_indices] = False
                choices_batched = choices_batched[mask]

        # Concatenate the results
        concatenated_candidates = torch.cat(candidate_list, dim=0)
        concatenated_acq_values = torch.cat(acq_value_list, dim=0)

        # Reshape to desired format [q, n_best, -1]
        final_shape = [q, n_best, concatenated_candidates.shape[-1]]
        concatenated_candidates = concatenated_candidates.view(final_shape)

        # Reset acq_func to previous X_pending state
        acq_function.set_X_pending(base_X_pending)

        return concatenated_candidates, concatenated_acq_values

    with torch.no_grad():
        acq_values = _split_batch_eval_acqf(
            acq_function=acq_function, X=choices_batched, max_batch_size=max_batch_size
        )
    sorted_indices = torch.argsort(acq_values, descending=True)[:n_best]
    best_candidates = choices_batched[sorted_indices]
    best_acq_values = acq_values[sorted_indices]

    return best_candidates, best_acq_values


def _split_batch_eval_acqf(
    acq_function: AcquisitionFunction, X: Tensor, max_batch_size: int
) -> Tensor:
    return torch.cat([acq_function(X_) for X_ in X.split(max_batch_size)])


### Custom surrogates ###
class SklearnSurrogate(Model):
    # Snippets copied from BayBe https://github.com/emdgroup/baybe

    def __init__(self, surrogate) -> None:
        """Use sklearn model for Botorch

        :param surrogate: fitted sklearn model
        :type surrogate: sklearn
        """
        super().__init__()
        self._surrogate = surrogate

    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        x = X.to(torch.float64)

        t_shape = x.shape[:-2]
        q_shape = x.shape[-2]
        x_fl = x.flatten(end_dim=-2)

        _mu, _si = self._surrogate.predict(x_fl.numpy(), return_std=True)

        mu = torch.from_numpy(_mu)
        si = torch.from_numpy(_si).pow(2)

        q_mu = torch.reshape(mu, t_shape + (q_shape,))
        q_si = torch.reshape(si, t_shape + (q_shape,))

        cova = torch.diag_embed(q_si)
        cova.add_(torch.eye(cova.shape[-1]) * 1e-9)

        dist = MultivariateNormal(q_mu, cova)

        return GPyTorchPosterior(dist)



class ForestSurrogate(Model):
    # Snippets copied from BayBe https://github.com/emdgroup/baybe
    def __init__(self, surrogate) -> None:
        """Use sklearn forest model for Botorch

        :param surrogate: fitted sklearn ensemble model
        :type surrogate: sklearn
        """
        super().__init__()
        self._surrogate = surrogate

    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        x = X.to(torch.float64)

        t_shape = x.shape[:-2]
        q_shape = x.shape[-2]
        x_fl = x.flatten(end_dim=-2)

        preds = np.array(
            [
                self._surrogate.estimators_[tree].predict(x_fl)
                for tree in range(self._surrogate.n_estimators)
            ]
        )

        _mu, _si = preds.mean(0), preds.std(0)

        mu = torch.from_numpy(_mu)
        si = torch.from_numpy(_si).pow(2)

        q_mu = torch.reshape(mu, t_shape + (q_shape,))
        q_si = torch.reshape(si, t_shape + (q_shape,))

        cova = torch.diag_embed(q_si)
        cova.add_(torch.eye(cova.shape[-1]) * 1e-9)

        dist = MultivariateNormal(q_mu, cova)

        return GPyTorchPosterior(dist)

    #def forward(self, x: Tensor) -> MultivariateNormal:
    #    if self.training:
    #        x = self.transform_inputs(x)
    #    mean_x = self.mean_module(x)
    #    covar_x = self.covar_module(x)
    #    return MultivariateNormal(mean_x, covar_x)


class XGBoostSurrogate(Model):
    def __init__(self, surrogate) -> None:
        """Use XGBoost model for Botorch

        :param surrogate: fitted XGBoost model
        :type surrogate: xgboost.XGBRegressor
        """
        super().__init__()
        self._surrogate = surrogate

    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        x_np = X.detach().cpu().numpy()  # Convert tensor to NumPy array

        if isinstance(self._surrogate, xgb.XGBRegressor):
            # Get number of trees
            n_trees = self._surrogate.get_booster().num_boosted_rounds()

            # Get the predictions from each tree
            preds = np.array(
                [
                    self._surrogate.predict(x_np, ntree_limit=i + 1)
                    for i in range(n_trees)
                ]
            )

            # Compute mean and standard deviation
            _mu = preds.mean(axis=0)
            _si = preds.std(axis=0)
        else:
            raise TypeError(
                "Unsupported model type for XGBoost surrogate. Expected xgb.XGBRegressor."
            )

        mu = torch.from_numpy(_mu).unsqueeze(-1)
        si = torch.from_numpy(_si).pow(2).unsqueeze(-1)

        dist = MultivariateNormal(mu, torch.diag_embed(si))

        return GPyTorchPosterior(dist)

# Rules of Acquisition

## Motivation
Inspired by the Star Trek universe and following the Ferengi's 3rd rule of acquisition - "Never spend more for an acquisition than you have to," and the 74th rule - "Knowledge equals profit," we introduce two batch selection strategies for cost-efficient BO to find a good cost and yield increase compromise.

<img src="rules_ferengi.png" width="40%" height="40%" />

## Introduction
**Bayesian Optimization (BO) taking batch cost into account.** 
In this revised approach to BO, we focus on optimizing chemical experiments not just for their potential improvement in yield over the previous iteration, but also for cost-efficiency for performing the experiments. Computationally simulated BO experiments result in selections that may overlook the varying costs of the chemicals involved. For instance, instead of acquiring a new substance chemists might first study a reaction under varying conditions that can easily be controlled, such as temperature. Not only will such experiments result in lower costs but also in a better informed posterior and higher confidence before acquiring new compounds.
Our modified approach adds a crucial dimension to the BO: the cost and ease of availability of each compound used at each batch iteration. Thus cost-informed BO will mimic more closely the yield optimization process in a chemistry lab.


## Installation

Python dependencies:

```
torch
gpytorch
botorch
rdkit
matplotlib
sklearn
numpy
```
Best to create a new envirnment and then

```
pip install -r requirements.txt
```


After setting up an envirnment with these packages, add
```
git clone git@github.com:janweinreich/rules_of_acquisition.git
export PYTHONPATH=$PYTHONPATH:$HOME/rules_of_acquisition
```
to your .bashrc file. Then, run
```
source ~/.bashrc
```
Currently tested with python 3.8.16. and botorch 0.8.1.


## Repository Structure

### `RegressionDemo`


Perform various regression tasks on the Pd-catalyzed C-H arylation dataset [1] resulting in a scatter plot with errorbars (`correlation.png`). 
All regressors are compatible with `botorch`: 

Gaussian Process Regression: `GPR.py` Try the effect of different kernels: `Tanimoto` kernel performs quite well and is the default choice. Other options include `RBF`, `Matern` and `Linear`.  Random Forest Regression: `ForestReg.py`. XgBoost Regression: `XgBoostREG.py`


### `AcqFuncPrice`


Use a modified acquisition function $\alpha_{p_j}^{i}$ with dimension (aquisition function/price) where $p_j$ is the current price of ligand $j$ and $i$ is the index of the batch.
The original acquisition function (here GIBBON by default) is not modified, but the values are divided by a monotonic increasing function of the price. 
This allows using different acquisition functions implemented in `botorch`.
Empirically we find a good choice for the acquisition function value associated to each experiment $i$ as:

$\alpha_{p}^{i} = \alpha / (1+\log(p_{j})) \text{ , if } j \text{ not in inventory}  $

If a ligand was already included (by buying 1 g of the substrance ) we divide by $1$. This does not requiere more user input that the price per ligand. After computing all $\alpha_{p}^{i}$ values the batches are reranked and the batch with the largest value

$\sum_{i} \alpha_{p}^{i}$

is selected.

### `FixBatch`

Corresponds to a greedy strategy where the user has to define a maximal cost for the batch using a config file `exp_configs.py`. For example setting `max_batch_cost=100` means per iteration `100` can be spend. If a suggested batch is more expensive, it is disregarded and the next best batch is just that can be afforded. 
More on the `exp_configs.py` below! If no batch can be afforded, take one where no new compunds are bought meaning measure difference temperatures or concentrations.

### `SI`

Contains all experiments that are shown in the SI of the paper.
Documentation for that is not up to date (see below).


### How do I control and select an experiment?


_____________

## Strategies

### Scan Maxima of Acquisition Function:
   - **Objective**: To maximize acquisition value within budget limits.
   - **Process**:
     - Begin with the best batch as per the acquisition function.
     - Sequentially evaluate affordability of subsequent batches.

### Greedy:
   - **Objective**: To find a feasible batch within budget constraints.
   - **Process**:


### `misc`
- **Content**: Space for experimental or outdated items.




### `1_greedy_fix_costs`
- **Description**: Implements the Greedy algorithm with fixed sample costs.
- **Subfolders**:
  - `values`: Focuses on ligands with the lowest yield across conditions.
  - `distance`: Divides dataset by proximity to the best ligand in feature space, starting with the furthest half.
       - Start with the desired batch size.
     - Increase batch size incrementally if the current batch is not affordable.
     - Subselect from suggested batches to fit the budget.

### `2_greedy_variable_costs`
- **Description**: Greedy algorithm considering variable costs. Acquiring a sample reduces its cost to zero for subsequent changes like temperature or solvent adjustments.
     - Start with the desired batch size.
     - Increase batch size incrementally if the current batch is not affordable.
     - Subselect from suggested batches to fit the budget.

### `4_similarity_based_costs`
- **Description**: Cost based on similarity to previously synthesized compounds. Assumes new compounds are synthesized, not purchased, with cost reflecting synthetic difficulty.


## Contributions
We welcome contributions and suggestions!


## :scroll: License
This project is licensed under the MIT License


## References
[1] Shields, B. J.; Stevens, J.; Li, J.; Parasram, M.; Damani, F.; Alvarado, J. I. M.; Janey, J. M.;
Adams, R. P.; Doyle, A. G. Nature 2021, 590, 89–96
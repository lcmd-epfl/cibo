# Rules of Acquisition

<img src="rules_ferengi.png" width="80%" height="80%" />

## Introduction
**Bayesian Optimization (BO) taking batch cost into account.** Inspired by the Star Trek universe, the "Rules of Acquisition" are a series of sacred business proverbs from the ultra-capitalist Ferengi race. This project embodies the principles of efficiency and knowledge in BO, particularly focusing on cost-aware batch selection strategies.

## Motivation
Following the Ferengi's 3rd rule of acquisition - "Never spend more for an acquisition than you have to," and the 74th rule - "Knowledge equals profit," we introduce two innovative batch selection strategies for cost-efficient BO.


## Installation

Add
```
git clone git@github.com:janweinreich/rules_of_acquisition.git
export PYTHONPATH=$PYTHONPATH:$HOME/rules_of_acquisition
```
to your .bashrc file. Then, run
```
source ~/.bashrc
```
Currently tested with python 3.8.16. and botorch 0.8.1.


## Strategies

### Scan Maxima of Acquisition Function:
   - **Objective**: To maximize acquisition value within budget limits.
   - **Process**:
     - Begin with the best batch as per the acquisition function.
     - Sequentially evaluate affordability of subsequent batches.

### Greedy Approach:
   - **Objective**: To find a feasible batch within budget constraints.
   - **Process**:
     - Start with the desired batch size.
     - Increase batch size incrementally if the current batch is unaffordable.
     - Subselect from suggested batches to fit the budget.



## Repository Structure

### `3_scan_opt_update_cost_acq_per_price`
- **Description**:
To find a good cost and yield increase compromise.
Use a modified acquisition function $\alpha_{p_j}^{i}$ with dimension (aquisition function/p) where $p_j$ is the current price of ligand $j$ and $i$ is the index of the batch. The acquisition function value associated to each experiment $i$ within a batch is now:

$\alpha_{p}^{i} = \alpha / (1+\log(p_{j})) \text{ , if } j \text{ not in inventory}  $

If a ligand was already included (by buying 1 g of the substrance ) we divide by $1$. This does not requiere more user input that the price per ligand. After computing all $\alpha_{p}^{i}$ values the batches are ranked according to 

$Batch value per money = \sum_{i} \alpha_{p}^{i}$


### `2_scan_opt_update_costs_saving`
- **Description**:
User has to define a maximal cost for the batch. If a suggested batch is more expensive, it is disregarded and the next best batch is just that can be afforded. If no batch can be afforded, take one where no new compunds are bought meaning measure difference temperatures or concentrations.

### `misc`
- **Content**: Space for experimental or outdated items.


### SI
Contains all experiments that are shown in the SI of the paper.

### `1_greedy_fix_costs`
- **Description**: Implements the Greedy algorithm with fixed sample costs.
- **Subfolders**:
  - `values`: Focuses on ligands with the lowest yield across conditions.
  - `distance`: Divides dataset by proximity to the best ligand in feature space, starting with the furthest half.

### `2_greedy_variable_costs`
- **Description**: Greedy algorithm considering variable costs. Acquiring a sample reduces its cost to zero for subsequent changes like temperature or solvent adjustments.

### `4_similarity_based_costs`
- **Description**: Cost based on similarity to previously synthesized compounds. Assumes new compounds are synthesized, not purchased, with cost reflecting synthetic difficulty.


## Contributions
We welcome contributions and suggestions!


## :scroll: License
This project is licensed under the MIT License
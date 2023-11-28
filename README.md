# Rules of Acquisition

<img src="rules_ferengi.png" width="80%" height="80%" />

## Introduction
**Bayesian Optimization (BO) made simple.** Inspired by the Star Trek universe, the "Rules of Acquisition" are a series of sacred business proverbs from the ultra-capitalist Ferengi race. This project embodies the principles of efficiency and knowledge in BO, particularly focusing on cost-aware batch selection strategies.

## Motivation
Following the Ferengi's 3rd rule of acquisition - "Never spend more for an acquisition than you have to," and the 74th rule - "Knowledge equals profit," we introduce two innovative batch selection strategies for cost-efficient BO.


## Installation

Add
```
export PYTHONPATH=$PYTHONPATH:$HOME/projects/rules_of_acquisition
```
to your .bashrc file. Then, run
```
source ~/.bashrc
```
Currently tested with python 3.8.16. and botorch 0.8.1.
Deepchem dependency will be removed in future versions.


## Strategies
### 1. Greedy Approach:
   - **Objective**: To find a feasible batch within budget constraints.
   - **Process**:
     - Start with the desired batch size.
     - Increase batch size incrementally if the current batch is unaffordable.
     - Subselect from suggested batches to fit the budget.
   - **Application**: Ideal for scenarios with fixed costs per sample.

### 2. Scan Maxima of Acquisition Function:
   - **Objective**: To maximize acquisition value within budget limits.
   - **Process**:
     - Begin with the best batch as per the acquisition function.
     - Sequentially evaluate affordability of subsequent batches.
   - **Application**: Suitable when cost varies based on batch composition.

## Repository Structure
### `1_greedy_fix_costs`
- **Description**: Implements the Greedy algorithm with fixed sample costs.
- **Subfolders**:
  - `values`: Focuses on ligands with the lowest yield across conditions.
  - `distance`: Divides dataset by proximity to the best ligand in feature space, starting with the furthest half.

### `2_greedy_variable_costs`
- **Description**: Greedy algorithm considering variable costs. Acquiring a sample reduces its cost to zero for subsequent changes like temperature or solvent adjustments.

### `3_similarity_based_costs`
- **Description**: Cost based on similarity to previously synthesized compounds. Assumes new compounds are synthesized, not purchased, with cost reflecting synthetic difficulty.

### `misc`
- **Usage**: Space for experimental or outdated items.

## Contributions
We welcome contributions and suggestions to improve the implementation and efficiency of these strategies. For contributing, please refer to the [contribution guidelines](CONTRIBUTING.md).

## License
This project is licensed under the [MIT License](LICENSE.md).
# Cost-Informed Bayesian Reaction Optimization (CIBO) a.k.a. Rules of Acquisition

## Motivation
_Inspired by the Star Trek universe following Ferengi's 3rd rule of acquisition - "Never spend more for an acquisition than you have to," and the 74th rule - "Knowledge equals profit," we introduce strategies for cost-efficient BO to find a good cost and yield increase compromise._

<img src="rules_ferengi.png" width="40%" height="40%" />

## Abstact
Bayesian optimization (BO) of reactions becomes increasingly important for advancing chemical discovery. Although effective in guiding experimental design, BO does not account for experimentation costs. For example, it may be more cost-effective to measure a reaction with the same ligand multiple times at different temperatures than buying a new one. We present Cost-Informed BO (CIBO), a policy tailored for chemical experimentation to prioritize experiments with lower costs. In contrast to BO, CIBO finds a cost-effective sequence of experiments towards the global optimum, the “mountain peak”. We envision use cases for efficient resource allocation in experimentation planning for traditional or self-driving laboratories.

<img src="cibo-toc_concept_3.png" width="70%" height="60%" />

## Introduction
**Bayesian Optimization (BO) taking batch cost into account.** 

Our modified approach adds a crucial dimension to the BO: the cost and ease of availability of each compound used at each batch iteration.


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
git clone git@github.com:lcmd-epfl/cibo.git
export PYTHONPATH=$PYTHONPATH:$HOME/rules_of_acquisition
```
to your `.bashrc` file. Then, run
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


If a ligand was already included (by buying 1 g of the substrance ) we divide by $1$. This does not requiere more user input that the price per ligand. After computing all $\alpha_{p}^{i}$ values the batches are reranked and the batch with the largest value

$\sum_{i} \alpha_{p}^{i}$

is selected.




### How do I control and select an experiment?


_____________

## Strategies


### `misc`
- **Content**: Space for experimental or outdated items.




## Contributions
We welcome contributions and suggestions!


## :scroll: License
This project is licensed under the MIT License


## References
[1] Shields, B. J.; Stevens, J.; Li, J.; Parasram, M.; Damani, F.; Alvarado, J. I. M.; Janey, J. M.;
Adams, R. P.; Doyle, A. G. Nature 2021, 590, 89–96
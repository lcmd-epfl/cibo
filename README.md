# Cost-Informed Bayesian Reaction Optimization (CIBO) a.k.a. Rules of Acquisition

Related preprint:

https://chemrxiv.org/engage/chemrxiv/article-details/66220e8a21291e5d1d27408d

Authors:

Alexandre A. Schoepfer, Jan Weinreich, Ruben Laplaza,Jerome
Waser, and Clemence Corminboeuf

## Motivation
_Inspired by the Star Trek universe following Ferengi's 3rd rule of acquisition - "Never spend more for an acquisition than you have to," and the 74th rule - "Knowledge equals profit," we introduce strategies for cost-efficient BO to find a good cost and yield increase compromise._

<img src="rules_ferengi.png" width="60%" height="40%" />





## Abstact
Bayesian optimization (BO) of reactions becomes increasingly important for advancing chemical discovery. Although effective in guiding experimental design, BO does not account for experimentation costs. For example, it may be more cost-effective to measure a reaction with the same ligand multiple times at different temperatures than buying a new one. We present Cost-Informed BO (CIBO), a policy tailored for chemical experimentation to prioritize experiments with lower costs. In contrast to BO, CIBO finds a cost-effective sequence of experiments towards the global optimum, the “mountain peak”. We envision use cases for efficient resource allocation in experimentation planning for traditional or self-driving laboratories.

<img src="cibo-toc_concept_3.png" width="60%" height="60%" />

CIBO vs BO. BO suggests a direct and steep path with expensive climbing equipment and a higher chance of costs for suffering injuries. CIBO suggests a slightly longer but safer path with lower equipment costs needed for the ascent.

## What problem are we solving?
Add a crucial dimension to the BO: the cost and ease of availability of each compound used at each batch iteration.
<img src="overview.png" width="100%" height="60%" />

Overview of standard BO (blue) vs. _cost-informed Bayesian optimization_ (CIBO, orange) for yield optimization.

 (a): BO recommends purchasing more materials. Meanwhile, CIBO balances purchases with their expected improvement of the experiment, at the cost of performing more experiments (here five vs. four). 

(b): A closer look at the two acquisition functions of BO and CIBO for the selection of experiment two. In CIBO, the BO acquisition function is modified to account for the cost by subtracting the latter. Following the blue BO curve, the next experiment to perform uses green and red reactants (corresponding to the costly maximum on the right). Subtracting the price of the experiments results in the orange CIBO curve, which instead suggests the more cost-effective experiment on the left (blue and red reactants).


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
Best to create a new environment (tested with python 3.8.16. and botorch 0.8.1)
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


## Repository Structure

### `data`
Currently supports two different datasets:
Direct arylation (DA) [1] and Cross-coupling (CC) [2] with yields ranging from 0–100%. To add your own dataset create a preprocessing script similar to `data/baumgartner.py` or `data/directaryl.py` and add the option to load your data to the `Evaluation_data` class in `data/datasets.py`
The datasets are called "BMS" and "baumgartner" respectively, as a keywork in the `exp_config.py` files.


### `RegressionDemo`

Regression on both datasets resulting in a scatter plot with errorbars (`correlation.png`). All regressors are compatible with `botorch`: 

Gaussian Process Regression: `GPR.py` Try the effect of different kernels: `Tanimoto` kernel performs quite well and is the default choice. Optionally also try Random Forest regression `RFR.py` interfaced with `sklearn`.

To change the dataset `"dataset"`, initialization scheme (`"init_strategy"`) and number of training points `"ntrain"` open the `exp_configs_1.py` file. The other keywords have no effect for these two scripts and are only relevant for the Bayesian optimization runs.

### `AcqFuncPrice`

Reproduce figures from the paper: for the two different datasets.




### `misc`
- **Content**: Space for experimental or outdated items.



## Contributions
We welcome contributions and suggestions!


## :scroll: License
This project is licensed under the MIT License


## References
[1] Shields, B. J.; Stevens, J.; Li, J.; Paras-
ram, M.; Damani, F.; Alvarado, J. I. M.;
Janey, J. M.; Adams, R. P.; Doyle, A. G.
Bayesian reaction optimization as a tool
for chemical synthesis. Nature 2021, 590,
89–96.

[2] Baumgartner, L. M.; Dennis, J. M.;
White, N. A.; Buchwald, S. L.;
Jensen, K. F. Use of a droplet plat-
form to optimize Pd-catalyzed C–N
coupling reactions promoted by organic
bases. Org. Process Res. Dev. 2019, 23,
1594–1601
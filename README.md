# Rules of Acquisition

Bayesian Optimization made simple. Name derived from Star Trek universe, the "Rules of Acquisition" are a collection of sacred business proverbs of the ultra-capitalist race known as the Ferengi.

We will follow the 3rd rule of aquisition "Never spend more for an acquisition than you have to." as well as rule 74 "Knowledge equals profit."

Structure of the folders:

- `1_greedy_fix_costs``: Greedy algorithm with fixed costs:
  We assume that the price of each sample is fixed:
  For instance if a sample was acquired previously and we want to re-use it at a different temperature we still pay the same price.

  Has two subfolders:
  - `values`: We start with all datapoints for the ligand that has the smallest yield over all 
  reaction conditions

  - `distance`: We divide the dataset into the half costest to the best ligand (in feature space) and the rest. We start with the half that is furthest away from the best ligand (in feature space that also includes temperature, solvent, etc.)

- `2_greedy_variable_costs`: Greedy algorithm with variable costs: After acquiring a sample, the price of the sample decreases to zero. E.g. we just change the temperature or the solvent (both are cheap or free).

- `3_similarity_based_costs`: We assume that the price of each sample is based on the similarity to the previous samples in the training set or the batch. Assumption is that each now compound is synthesized not bought. The price is based on the similarity to the previous compounds. The more similar the cheaper the compound is. The similarity should reflect syntehtic difficulty: For instance two similar molecules are similarly hard to make. If a molecule was made before (it is in the training set) it is free to make again.


- misc: there you can put stuff to experiment or outdated things

<img src="rules.png" width="50%" height="50%" />

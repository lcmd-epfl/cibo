import pandas as pd
import numpy as np
import math


def buchwald_prices():
    # load the Buchwald_prices.csv
    price_data = pd.read_csv(
        "https://raw.githubusercontent.com/janweinreich/rules_of_acquisition/main/data/ahneman2018_prices.csv"
    )
    # create a price dictionary for each compound type
    price_dict_additives = {}
    price_dict_aryl_halide = {}
    price_dict_base = {}
    price_dict_ligand = {}

    # select all additives from  the price_data
    additives = price_data[price_data.type == "additive"]
    aryl_halides = price_data[price_data.type == "aryl_halide"]
    bases = price_data[price_data.type == "base"]
    ligands = price_data[price_data.type == "ligand"]

    for smiles, cost in zip(additives.smiles, additives.cost_per_gram):
        price_dict_additives[smiles] = cost

    price_dict_additives["zero"] = 0.0

    for smiles, cost in zip(aryl_halides.smiles, aryl_halides.cost_per_gram):
        price_dict_aryl_halide[smiles] = cost

    price_dict_aryl_halide["zero"] = 0.0

    for smiles, cost in zip(bases.smiles, bases.cost_per_gram):
        price_dict_base[smiles] = cost

    for smiles, cost in zip(ligands.smiles, ligands.cost_per_gram):
        price_dict_ligand[smiles] = cost

    return (
        price_dict_additives,
        price_dict_aryl_halide,
        price_dict_base,
        price_dict_ligand,
    )


if __name__ == "__main__":
    (
        price_dict_additives,
        price_dict_aryl_halide,
        price_dict_base,
        price_dict_ligand,
    ) = buchwald_prices()

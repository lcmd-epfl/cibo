import pandas as pd

def baumgartner_split_nucleophiles():
    data = pd.read_csv("baumgartner2019_reaction_data.csv")
    # Unique N-H nucleophiles
    nucleophiles = ["Aniline", "Benzamide", "Morpholine", "Phenethylamine"]
    # Create a dictionary of dataframes, each containing data for a specific nucleophile
    dataframes = {}
    for nucleophile in nucleophiles:
        dataframes[nucleophile] = data[data["N-H nucleophile "] == nucleophile]

    return dataframes


def baumgartner2019_prices():
    price_data = pd.read_csv("baumgartner2019_compound_info.csv")

    price_dict_precatalyst = {}
    price_dict_solvent = {}
    price_dict_base = {}

    # select all additives from  the price_data
    precatalyst = price_data[price_data.type == "precatalyst"]
    solvent = price_data[price_data.type == "make-up solvent"]
    base = price_data[price_data.type == "base"]

    for smiles, cost in zip(precatalyst.smiles, precatalyst.cost_per_gram):
        price_dict_precatalyst[smiles] = cost

    for smiles, cost in zip(solvent.smiles, solvent.cost_per_gram):
        price_dict_solvent[smiles] = cost

    for smiles, cost in zip(base.smiles, base.cost_per_gram):
        price_dict_base[smiles] = cost

    return price_dict_precatalyst, price_dict_solvent, price_dict_base

import pandas as pd

def baumgartner2019_prices():
    price_data = pd.read_csv(
        "https://raw.githubusercontent.com/janweinreich/rules_of_acquisition/main/data/baumgartner2019_compound_info.csv"
    )

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

    all_price_dicts = {
        "precatalyst": price_dict_precatalyst,
        "solvent": price_dict_solvent,
        "base": price_dict_base,
    }

    name_smiles_dict = {}
    for name, smiles in zip(price_data.name, price_data.smiles):
        name_smiles_dict[name] = smiles

    data = pd.read_csv(
        "https://raw.githubusercontent.com/janweinreich/rules_of_acquisition/main/data/baumgartner2019_reaction_data.csv"
    )
    # Unique N-H nucleophiles
    # round yield above 100 to 100
    data["yield"] = data["Reaction Yield"].apply(lambda x: 100.0 if x > 100.0 else x)

    precatalyst_smiles = []
    solvent_smiles = []
    base_smiles = []
    for precatalyst, solvent, base in zip(
        data["Precatalyst"], data["Make-Up Solvent ID"], data["Base"]
    ):
        precatalyst_smiles.append(name_smiles_dict[precatalyst])
        solvent_smiles.append(name_smiles_dict[solvent])
        base_smiles.append(name_smiles_dict[base])

    data["precatalyst_smiles"] = precatalyst_smiles
    data["solvent_smiles"] = solvent_smiles
    data["base_smiles"] = base_smiles

    nucleophiles = ["Aniline", "Benzamide", "Morpholine", "Phenethylamine"]
    # Create a dictionary of dataframes, each containing data for a specific nucleophile
    dataframes = {}
    for nucleophile in nucleophiles:
        dataframes[nucleophile] = data[data["N-H nucleophile "] == nucleophile]

    return dataframes, all_price_dicts


if __name__ == "__main__":
    print(baumgartner2019_prices())

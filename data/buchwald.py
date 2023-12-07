import pandas as pd
import pdb
import numpy as np

# load the Buchwald_prices.csv

data = pd.read_csv("Buchwald_prices.csv")

chem_types = data["type"].unique()


for sub in chem_types:
    sub_data = data[data["type"] == sub]
    # Calculate the mean cost for the current sub category
    mean_cost = sub_data["cost"].mean()
    # Fill NaN values with the mean cost for the current sub category
    data.loc[data["type"] == sub, "cost"] = sub_data["cost"].fillna(mean_cost)

# Save the new data
data.to_csv("Buchwald_prices_filled_nan.csv", index=False)
pdb.set_trace()
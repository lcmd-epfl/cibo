import pandas as pd
import numpy as np
import pdb

extended_molecule_emojis = [
    '🌟',  # Star
    '💎',  # Gem Stone
    '💡',  # Light Bulb
    '🎈',  # Balloon
    '🧪',  # Test Tube
    '⚗️',  # Alembic
    '🧫',  # Petri Dish
    '🧬',  # DNA Double Helix
    '🔬',  # Microscope
    '🧲',  # Magnet
    '⚛️',  # Atom symbol
    '🌡️',  # Thermometer
    '💊',  # Pill
    '🔭',  # Telescope
    '🌌',  # Galaxy (for complex molecular structures)
    '💧',  # Droplet (for liquid molecules)
    '🍃',  # Leaf (for organic molecules)
    '🔋',  # Battery (for energy-related molecules)
    '💣',  # Bomb (for reactive or explosive molecules)
    '⚖️'   # Balance Scale (for measuring and balancing chemical equations)
]



log_data_normal = pd.read_csv("./DirectAryl/exp_log_normal.csv")
log_data_coi = pd.read_csv("./DirectAryl/exp_log_coi.csv")
# extract Ligand and Iteration columns
log_data_normal = np.array(log_data_normal[["Ligand"]].values.reshape((30, 5)))
log_data_coi = np.array(log_data_coi[["Ligand"]].values.reshape((30, 5)))
UNIQUE_LIGANDS = np.unique(log_data_normal.flatten())


molecule_emojis = extended_molecule_emojis[: len(UNIQUE_LIGANDS)]

# create a dictionary of ligands and emojis
ligand_emoji_dict = dict(zip(UNIQUE_LIGANDS, molecule_emojis))


bought_normal = []
bought_coi = []
print("Normal|COI")
for i in range(30):
    print(f"IT {i+1}----")
    for j in range(5):
        normal_stuff = ligand_emoji_dict[log_data_normal[i][j]]
        coi_stuff = ligand_emoji_dict[log_data_coi[i][j]]

        if normal_stuff not in bought_normal:
            bought_normal.append(normal_stuff)
        if coi_stuff not in bought_coi:
            bought_coi.append(coi_stuff)

        print(f"{normal_stuff}|{coi_stuff}")

    print(len(bought_normal), len(bought_coi))

print(ligand_emoji_dict)

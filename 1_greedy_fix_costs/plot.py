import matplotlib.pyplot as plt
import numpy as np
from utils import *
import pdb
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator


def generate_color_scale(iterations, cmap_name='coolwarm'):
    cmap = plt.get_cmap(cmap_name)
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, iterations)]


REACTTION_1 = "ebdo_direct_arylation"
REACTTION_2 = "buchwald"
RESULTS = load_pkl("results_4_11.pkl")


random_results_ebdo = np.mean(np.array(RESULTS[0]["y_better_RANDOM_ALL"]), axis=0)
all_bo_results_ebdo = []
for i in range(6):
    bo_results = np.mean(np.array(RESULTS[i]["y_better_BO_ALL"]), axis=0)
    all_bo_results_ebdo.append(bo_results)

all_bo_results_ebdo = np.array(all_bo_results_ebdo)

random_results_buchwald = np.mean(np.array(RESULTS[6]["y_better_RANDOM_ALL"]), axis=0)
all_bo_results_buchwald = []
for i in range(6, 12):
    bo_results = np.mean(np.array(RESULTS[i]["y_better_BO_ALL"]), axis=0)
    all_bo_results_buchwald.append(bo_results)

all_bo_results_buchwald = np.array(all_bo_results_buchwald)
plt.style.use('seaborn-poster')  # Apply a global aesthetic style.
# Plot
# Increased figure size for clarity.
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

# EBDO Direct Arylation
ax[0].plot(random_results_ebdo, label="Random", color="black", ls="--")
for i in range(6):
    ax[0].plot(all_bo_results_ebdo[i],
               label=f"Max $ per Batch: {i}", color=generate_color_scale(6)[i])
# Force integer x-axis labels.
ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Yield (%)")  # Assuming yield is in percentage.
ax[0].set_title("EBDO Direct Arylation")
ax[0].legend()

# Adjusting spines for a cleaner look, keep bottom and left spines visible
for position in ['top', 'right']:
    ax[0].spines[position].set_visible(False)

# Buchwald-Hartwig Amination (assuming this is the reaction)
ax[1].plot(random_results_buchwald, label="Random", color="black", ls="--")
for i in range(6):
    ax[1].plot(all_bo_results_buchwald[i],
               label=f"Max $ per Batch: {i}", color=generate_color_scale(6)[i])
# Force integer x-axis labels.
ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Yield (%)")  # Assuming yield is in percentage.
ax[1].set_title("Buchwald-Hartwig Amination")
ax[1].legend()

# Adjusting spines for a cleaner look, keep bottom and left spines visible
for position in ['top', 'right']:
    ax[1].spines[position].set_visible(False)

plt.tight_layout()
plt.savefig("results_4_11.png")
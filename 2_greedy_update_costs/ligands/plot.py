import matplotlib.pyplot as plt
import numpy as np
from utils import *
import pdb
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator


def generate_color_scale(iterations, cmap_name='coolwarm'):
    cmap = plt.get_cmap(cmap_name)
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, iterations)]



#running_costs_RANDOM_ALL
#running_costs_BO_ALL


ITERATIONS = np.arange(1, 22)

REACTTION_1 = "ebdo_direct_arylation"
RESULTS = load_pkl("results_5_11.pkl")


random_results_ebdo = np.mean(np.array(RESULTS[0]["y_better_RANDOM_ALL"]), axis=0)
random_costs_ebdo   = np.mean(np.array(RESULTS[0]["running_costs_RANDOM_ALL"]), axis=0)

all_bo_results_ebdo = []
all_bo_costs_ebdo   = []
for i in range(6):
    bo_results = np.mean(np.array(RESULTS[i]["y_better_BO_ALL"]), axis=0)
    bo_costs   = np.mean(np.array(RESULTS[i]["running_costs_BO_ALL"]), axis=0)
    all_bo_results_ebdo.append(bo_results)
    all_bo_costs_ebdo.append(bo_costs)

all_bo_results_ebdo = np.array(all_bo_results_ebdo)
all_bo_costs_ebdo   = np.array(all_bo_costs_ebdo)




plt.style.use('seaborn-poster')  # Apply a global aesthetic style.
# Plot
# Increased figure size for clarity.
fig, ax = plt.subplots(2, 1, figsize=(7, 7))

# EBDO Direct Arylation
ax[0].plot(ITERATIONS, random_results_ebdo, label="Random", color="black", ls="--")
ax[1].plot(ITERATIONS, random_costs_ebdo, label="Random", color="black", ls="--")
for i in range(6):
    ax[0].plot(ITERATIONS, all_bo_results_ebdo[i],
               label=f"Max $ per Batch: {i}", color=generate_color_scale(6)[i])
    ax[1].plot(ITERATIONS, all_bo_costs_ebdo[i],
               label=f"Max $ per Batch: {i}", color=generate_color_scale(6)[i])
    

# Force integer x-axis labels.
ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
#ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Yield [%]")  # Assuming yield is in percentage.
ax[0].set_title("EBDO Direct Arylation")
ax[0].legend(loc="lower right", fontsize=12)


ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Sum(Costs) [$]")




# Adjusting spines for a cleaner look, keep bottom and left spines visible
for position in ['top', 'right']:
    ax[0].spines[position].set_visible(False)
    ax[1].spines[position].set_visible(False)


#MAKE X LIM (1,13) everywhere
ax[0].set_xlim(0.5, 13)
ax[1].set_xlim(0.5, 13)

plt.tight_layout()
plt.savefig("results_5_11.png")
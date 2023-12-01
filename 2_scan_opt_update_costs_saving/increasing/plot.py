import matplotlib.pyplot as plt
import numpy as np
from utils import *
import pdb
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["figure.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2


def generate_color_scale(iterations, cmap_name="coolwarm"):
    cmap = plt.get_cmap(cmap_name)
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, iterations)]


ITERATIONS = np.arange(16) + 1
RESULTS = load_pkl("results.pkl")


random_results = np.mean(np.array(RESULTS[0]["y_better_RANDOM_ALL"]), axis=0)
random_costs = np.mean(np.array(RESULTS[0]["running_costs_RANDOM_ALL"]), axis=0)


bo_results_aware = np.mean(np.array(RESULTS[0]["y_better_BO_ALL"]), axis=0)
bo_costs_aware = np.mean(np.array(RESULTS[0]["running_costs_BO_ALL"]), axis=0)

bo_results_unaware = np.mean(np.array(RESULTS[3]["y_better_BO_ALL"]), axis=0)
bo_costs_unaware = np.mean(np.array(RESULTS[3]["running_costs_BO_ALL"]), axis=0)

plt.style.use("seaborn-poster")  # Apply a global aesthetic style.

fig1, ax1 = plt.subplots(2, 1, figsize=(7, 7))

ax1[0].plot(
    ITERATIONS, bo_results_unaware, label="BO", color="red", marker="o", ls="--"
)
ax1[0].plot(
    ITERATIONS, bo_results_aware, label="BO-COST", color="navy", marker="o", ls="-."
)

ax1[1].plot(ITERATIONS, bo_costs_unaware, label="BO", color="red", marker="o", ls="--")
ax1[1].plot(
    ITERATIONS, bo_costs_aware, label="BO-COST", color="navy", marker="o", ls="-."
)
# make x axis ticks integers
ax1[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax1[1].xaxis.set_major_locator(MaxNLocator(integer=True))


ax1[0].set_ylabel("Yield [%]")  # Assuming yield is in percentage.
ax1[1].set_ylabel("Sum(Cost) [$]")
ax1[1].set_xlabel("Iteration")
ax1[0].legend(loc="lower right", fontsize=13)
ax1[1].legend(loc="lower right", fontsize=13)

# make tight
plt.tight_layout()

plt.savefig("optimization.png")
plt.close()

fig2, ax2 = plt.subplots(1, 1, figsize=(7, 7))


ax2.plot(
    random_costs, random_results, label="Random", color="black", marker="o", ls="-"
)
ax2.plot(
    bo_costs_aware, bo_results_aware, label="BO-COST", color="navy", marker="o", ls="-."
)
ax2.plot(
    bo_costs_unaware, bo_results_unaware, label="BO", color="red", marker="o", ls="--"
)

ax2.set_xlabel("Sum(Cost) [$]")
ax2.set_ylabel("Yield [%]")  # Assuming yield is in percentage.
ax2.legend(loc="lower right", fontsize=13)
# make tight
plt.tight_layout()
plt.savefig("comparison.png")


pdb.set_trace()
exit()


fig, ax = plt.subplots(2, 2, figsize=(14, 7))

# EBDO Direct Arylation
ax[0][0].plot(ITERATIONS, random_results_ebdo, label="Random", color="black", ls="--")
for i in range(6):
    ax[0][0].plot(
        ITERATIONS,
        all_bo_results_ebdo[i],
        label=f"Max $ per Batch: {i}",
        color=generate_color_scale(6)[i],
    )
# Force integer x-axis labels.
ax[0][0].xaxis.set_major_locator(MaxNLocator(integer=True))

ax[0][0].set_ylabel("Yield [%]")  # Assuming yield is in percentage.
ax[0][0].set_title("EBDO Direct Arylation")
ax[0][0].legend(loc="lower right", fontsize=13)

# Adjusting spines for a cleaner look, keep bottom and left spines visible
for position in ["top", "right"]:
    ax[0][0].spines[position].set_visible(False)

# Buchwald-Hartwig Amination (assuming this is the reaction)
ax[0][1].plot(
    ITERATIONS, random_results_buchwald, label="Random", color="black", ls="--"
)
for i in range(6):
    ax[0][1].plot(
        ITERATIONS,
        all_bo_results_buchwald[i],
        label=f"Max $ per Batch: {i}",
        color=generate_color_scale(6)[i],
    )
# Force integer x-axis labels.
ax[0][1].xaxis.set_major_locator(MaxNLocator(integer=True))

ax[0][1].set_ylabel("Yield [%]")  # Assuming yield is in percentage.
ax[0][1].set_title("Buchwald-Hartwig Amination")


# Adjusting spines for a cleaner look, keep bottom and left spines visible
for position in ["top", "right"]:
    ax[0][1].spines[position].set_visible(False)


ax[1][0].plot(ITERATIONS, random_costs_ebdo, label="Random", color="black", ls="--")

for i in range(6):
    ax[1][0].plot(
        ITERATIONS,
        all_bo_costs_ebdo[i],
        label=f"Max $ per Batch: {i}",
        color=generate_color_scale(6)[i],
    )

# Force integer x-axis labels.
ax[1][0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[1][0].set_xlabel("Iteration")
ax[1][0].set_ylabel("Sum(Cost) [$]")

# Adjusting spines for a cleaner look, keep bottom and left spines visible
for position in ["top", "right"]:
    ax[1][0].spines[position].set_visible(False)

ax[1][1].plot(ITERATIONS, random_costs_buchwald, label="Random", color="black", ls="--")

for i in range(6):
    ax[1][1].plot(
        ITERATIONS,
        all_bo_costs_buchwald[i],
        label=f"Max $ per Batch: {i}",
        color=generate_color_scale(6)[i],
    )

# Force integer x-axis labels.
ax[1][1].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[1][1].set_xlabel("Iteration")
ax[1][1].set_ylabel("Sum(Cost) [$]")


# MAKE X LIM (1,13) everywhere
# MAKE X LIM (1,13) everywhere
ax[0][0].set_xlim(0.5, 13)
ax[0][1].set_xlim(0.5, 13)
ax[1][0].set_xlim(0.5, 13)
ax[1][1].set_xlim(0.5, 13)

plt.tight_layout()
plt.savefig("results_4_11.png")

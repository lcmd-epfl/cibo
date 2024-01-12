import matplotlib.pyplot as plt
import numpy as np
from utils import *
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2


def generate_color_scale(iterations, cmap_name="coolwarm"):
    cmap = plt.get_cmap(cmap_name)
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, iterations)]


ITERATIONS = np.arange(31) + 1
RESULTS = load_pkl("results_DirectAryl.pkl")


BO_COA_YIELD = np.mean(np.array(RESULTS[0]["y_better_BO_ALL"]), axis=0)
BO_COA_COSTS = np.mean(np.array(RESULTS[0]["running_costs_BO_ALL"]), axis=0)

BO_YIELD = np.mean(np.array(RESULTS[1]["y_better_BO_ALL"]), axis=0)
BO_COSTS = np.mean(np.array(RESULTS[1]["running_costs_BO_ALL"]), axis=0)

pdb.set_trace()
plt.style.use("seaborn-poster")  # Apply a global aesthetic style.

fig1, ax1 = plt.subplots(2, 1, figsize=(7, 7))

ax1[0].plot(ITERATIONS, BO_COA_YIELD, label="BO-COA", color="red", marker="o", ls="--", alpha=0.5)
ax1[1].plot(ITERATIONS, BO_COA_COSTS, label="BO-COA", color="red", marker="o", ls="--", alpha=0.5)

ax1[0].plot(ITERATIONS, BO_YIELD, label="BO", color="blue", marker="o", ls="--", alpha=0.5)
ax1[1].plot(ITERATIONS, BO_COSTS, label="BO", color="blue", marker="o", ls="--", alpha=0.5)


# make x axis ticks integers
ax1[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax1[1].xaxis.set_major_locator(MaxNLocator(integer=True))


ax1[0].set_ylabel("Yield [%]")  # Assuming yield is in percentage.
ax1[1].set_ylabel("Sum(Cost) [$]")
ax1[1].set_xlabel("Iteration")
ax1[0].legend(loc="lower right", fontsize=13)
# ax1[1].legend(loc="lower right", fontsize=13)

# make tight
ax1[0].set_xlim([1, 20])
ax1[1].set_xlim([1, 20])
plt.tight_layout()

plt.savefig("DirectArylation.pdf")
plt.close()


RESULTS = load_pkl("results_Baumgartner.pkl")


fig2, ax2 = plt.subplots(2, 4, figsize=(14, 7))

BO_YIELD_Aniline = np.mean(np.array(RESULTS[0]["y_better_BO_ALL"]), axis=0)
BO_COSTS_Aniline = np.mean(np.array(RESULTS[0]["running_costs_BO_ALL"]), axis=0)
BO_COA_YIELD_Aniline = np.mean(np.array(RESULTS[1]["y_better_BO_ALL"]), axis=0)
BO_COA_COSTS_Aniline = np.mean(np.array(RESULTS[1]["running_costs_BO_ALL"]), axis=0)


BO_YIELD_Morpholine = np.mean(np.array(RESULTS[2]["y_better_BO_ALL"]), axis=0)
BO_COSTS_Morpholine = np.mean(np.array(RESULTS[2]["running_costs_BO_ALL"]), axis=0)
BO_COA_YIELD_Morpholine = np.mean(np.array(RESULTS[3]["y_better_BO_ALL"]), axis=0)
BO_COA_COSTS_Morpholine = np.mean(np.array(RESULTS[3]["running_costs_BO_ALL"]), axis=0)

BO_YIELD_Phenethylamine = np.mean(np.array(RESULTS[4]["y_better_BO_ALL"]), axis=0)
BO_COSTS_Phenethylamine = np.mean(np.array(RESULTS[4]["running_costs_BO_ALL"]), axis=0)
BO_COA_YIELD_Phenethylamine = np.mean(np.array(RESULTS[5]["y_better_BO_ALL"]), axis=0)
BO_COA_COSTS_Phenethylamine = np.mean(
    np.array(RESULTS[5]["running_costs_BO_ALL"]), axis=0
)

BO_YIELD_Benzamide = np.mean(np.array(RESULTS[6]["y_better_BO_ALL"]), axis=0)
BO_COSTS_Benzamide = np.mean(np.array(RESULTS[6]["running_costs_BO_ALL"]), axis=0)
BO_COA_YIELD_Benzamide = np.mean(np.array(RESULTS[7]["y_better_BO_ALL"]), axis=0)
BO_COA_COSTS_Benzamide = np.mean(np.array(RESULTS[7]["running_costs_BO_ALL"]), axis=0)


ax2[0, 0].set_title("Aniline", fontweight="bold")

ax2[0][0].set_ylabel("Yield [%]")  # Assuming yield is in percentage.

ax2[0, 0].plot(
    np.arange(25) + 1,
    BO_YIELD_Aniline,
    label="BO",
    color="blue",
    marker="o",
    ls="--",
    alpha=0.5,
)


ax2[0, 0].plot(
    np.arange(25) + 1,
    BO_COA_YIELD_Aniline,
    label="BO-COA",
    color="red",
    marker="o",
    ls="--",
    alpha=0.5,
)

ax2[0,0].legend(loc="lower right", fontsize=18, frameon=False)


ax2[1, 0].set_ylabel("Sum(Cost) [$]")
ax2[1, 0].plot(
    np.arange(25) + 1,
    BO_COSTS_Aniline,
    label="BO",
    color="blue",
    marker="o",
    ls="--",
    alpha=0.5,
)

ax2[1, 0].plot(
    np.arange(25) + 1,
    BO_COA_COSTS_Aniline,
    label="BO-COA",
    color="red",
    marker="o",
    ls="--",
    alpha=0.5,
)

ax2[0, 1].set_title("Morpholine", fontweight="bold")


ax2[0, 1].plot(
    np.arange(22) + 1,
    BO_YIELD_Morpholine,
    label="BO",
    color="blue",
    marker="o",
    ls="--",
    alpha=0.5,
)

ax2[0, 1].plot(
    np.arange(22) + 1,
    BO_COA_YIELD_Morpholine,
    label="BO-COA",
    color="red",
    marker="o",
    ls="--",
    alpha=0.5,
)

ax2[1, 1].plot(
    np.arange(22) + 1,
    BO_COSTS_Morpholine,
    label="BO",
    color="blue",
    marker="o",
    ls="--",
    alpha=0.5,
)

ax2[1, 1].plot(
    np.arange(22) + 1,
    BO_COA_COSTS_Morpholine,
    label="BO-COA",
    color="red",
    marker="o",
    ls="--",
    alpha=0.5,
)

ax2[0, 2].set_title("Phenethylamine", fontweight="bold")

ax2[0, 2].plot(
    np.arange(25) + 1,
    BO_YIELD_Phenethylamine,
    label="BO",
    color="blue",
    marker="o",
    ls="--",
    alpha=0.5,
)

ax2[0, 2].plot(
    np.arange(25) + 1,
    BO_COA_YIELD_Phenethylamine,
    label="BO-COA",
    color="red",
    marker="o",
    ls="--",
    alpha=0.5,
)


ax2[1, 2].plot(
    np.arange(25) + 1,
    BO_COSTS_Phenethylamine,
    label="BO",
    color="blue",
    marker="o",
    ls="--",
    alpha=0.5,
)

ax2[1, 2].plot(
    np.arange(25) + 1,
    BO_COA_COSTS_Phenethylamine,
    label="BO-COA",
    color="red",
    marker="o",
    ls="--",
    alpha=0.5,
)

ax2[0, 3].set_title("Benzamide", fontweight="bold")


ax2[0, 3].plot(
    np.arange(25) + 1,
    BO_YIELD_Benzamide,
    label="BO",
    color="blue",
    marker="o",
    ls="--",
    alpha=0.5,
)

ax2[0, 3].plot(
    np.arange(25) + 1,
    BO_COA_YIELD_Benzamide,
    label="BO-COA",
    color="red",
    marker="o",
    ls="--",
    alpha=0.5,
)

ax2[1, 3].plot(
    np.arange(25) + 1,
    BO_COSTS_Benzamide,
    label="BO",
    color="blue",
    marker="o",
    ls="--",
    alpha=0.5,
)

ax2[1, 3].plot(
    np.arange(25) + 1,
    BO_COA_COSTS_Benzamide,
    label="BO-COA",
    color="red",
    marker="o",
    ls="--",
    alpha=0.5,
)

ax2[1, 0].set_xlabel("Iteration")
ax2[1, 1].set_xlabel("Iteration")
ax2[1, 2].set_xlabel("Iteration")
ax2[1, 3].set_xlabel("Iteration")


# set ax limits for all ax in ax2
for i in range(2):
    for j in range(4):
        ax2[i, j].set_xlim([0, 20])

plt.tight_layout()
plt.savefig("Baumgartner.pdf")

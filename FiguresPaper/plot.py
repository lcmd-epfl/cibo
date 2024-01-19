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


BO_COI_YIELD = np.mean(np.array(RESULTS[0]["y_better_BO_ALL"]), axis=0)
BO_COI_COSTS = np.mean(np.array(RESULTS[0]["running_costs_BO_ALL"]), axis=0)

BO_YIELD = np.mean(np.array(RESULTS[1]["y_better_BO_ALL"]), axis=0)
BO_COSTS = np.mean(np.array(RESULTS[1]["running_costs_BO_ALL"]), axis=0)


plt.style.use("seaborn-poster")  # Apply a global aesthetic style.

fig1, ax1 = plt.subplots(2, 1, figsize=(7, 7))

ax1[0].plot(
    ITERATIONS,
    BO_COI_YIELD,
    label="BO-COI",
    color="red",
    marker="o",
    ls="--",
    alpha=0.5,
)
ax1[1].plot(
    ITERATIONS,
    BO_COI_COSTS,
    label="BO-COI",
    color="red",
    marker="o",
    ls="--",
    alpha=0.5,
)

ax1[0].plot(
    ITERATIONS, BO_YIELD, label="BO", color="blue", marker="o", ls="--", alpha=0.5
)
ax1[1].plot(
    ITERATIONS, BO_COSTS, label="BO", color="blue", marker="o", ls="--", alpha=0.5
)


# make x axis ticks integers
ax1[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax1[1].xaxis.set_major_locator(MaxNLocator(integer=True))


ax1[0].set_ylabel("Yield [%]")  # Assuming yield is in percentage.
ax1[1].set_ylabel(r"$\sum \rm cost ~ [\$]$")
ax1[1].set_xlabel("Iteration")
ax1[0].legend(loc="lower right", fontsize=13)
# ax1[1].legend(loc="lower right", fontsize=13)

# make tight
ax1[0].set_xlim([1, 20])
ax1[1].set_xlim([1, 20])
plt.tight_layout()

plt.savefig("DirectArylation.pdf")
plt.close()


RESULTS = load_pkl("results_Baumgartner_cheapest.pkl")


fig2, ax2 = plt.subplots(2, 4, figsize=(14, 7))
nulceophiles = [
    "Benzamide",
    "Phenethylamine",
    "Morpholine",
]
# plt.plot()
plt.show()
# pdb.set_trace()

for i, j in zip([0, 2, 4, 6], [0, 1, 2, 3]):
    nucleophile = RESULTS[i]["settings"]["nucleophile"]
    BO_YIELD = np.mean(np.array(RESULTS[i]["y_better_BO_ALL"]), axis=0)
    BO_COSTS = np.mean(np.array(RESULTS[i]["running_costs_BO_ALL"]), axis=0)

    BO_COA_YIELD = np.mean(np.array(RESULTS[i + 1]["y_better_BO_ALL"]), axis=0)
    BO_COA_COSTS = np.mean(np.array(RESULTS[i + 1]["running_costs_BO_ALL"]), axis=0)

    if nucleophile == "Morpholine":
        BO_COSTS += 548 + 2.8
        BO_COA_COSTS += 548 + 2.8
    else:
        BO_COSTS += 65.2 + 2.8
        BO_COA_COSTS += 65.2 + 2.8

    ITERATIONS = np.arange(len(BO_YIELD)) + 1

    ax2[0, j].set_title(f"{nucleophile}", fontweight="bold")

    if nucleophile != "Aniline":
        ax2[0, j].plot(
            ITERATIONS,
            BO_YIELD,
            label="BO",
            color="blue",
            marker="o",
            ls="--",
            alpha=0.5,
        )

        ax2[0, j].plot(
            ITERATIONS,
            BO_COA_YIELD,
            label="BO-COI",
            color="red",
            marker="o",
            ls="--",
            alpha=0.5,
        )

        # ax2[1, j].set_ylabel(r"$\sum \rm cost ~ [\$]$")
        ax2[1, j].plot(
            ITERATIONS,
            BO_COSTS,
            label="BO",
            color="blue",
            marker="o",
            ls="--",
            alpha=0.5,
        )

        ax2[1, j].plot(
            ITERATIONS,
            BO_COA_COSTS,
            label="BO-COI",
            color="red",
            marker="o",
            ls="--",
            alpha=0.5,
        )
    for k in range(2):
        ax2[k, j].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2[k, j].set_xlim([0, len(ITERATIONS) + 1])
        # ticsk
        ax2[k, j].set_xticks(np.arange(0, len(ITERATIONS) + 1, 5))
        ax2[k, j].tick_params(axis="both", which="major", labelsize=18)

ax2[0][0].set_ylabel("Yield [%]")  # Assuming yield is in percentage.
ax2[0, 0].legend(loc="lower right", fontsize=18, frameon=False)
ax2[1, 0].set_xlabel("Iteration")
ax2[1, 1].set_xlabel("Iteration")
ax2[1, 2].set_xlabel("Iteration")
ax2[1, 3].set_xlabel("Iteration")


RESULTS_ANILINE = load_pkl("results_Baumgartner_worst_Aniline.pkl")

"""
'CC(C)C1=CC(=C(C(=C1)C(C)C)C2=C(C(=CC=C2)OC(C)C)P(C3CCCCC3)C4CCCCC4)C(C)C'
(Pdb) worst_bases
*** NameError: name 'worst_bases' is not defined
(Pdb) self.worst_bases
'CCN(CC)CC'

(Pdb) self.price_dict_precatalyst[self.worst_precatalyst]
459.0
(Pdb) self.price_dict_base[self.worst_bases]
0.5730027548

"""
i, j = 0, 0
nucleophile = RESULTS_ANILINE[i]["settings"]["nucleophile"]


BO_YIELD = np.mean(np.array(RESULTS_ANILINE[i]["y_better_BO_ALL"]), axis=0)
BO_COSTS = (
    np.mean(np.array(RESULTS_ANILINE[i]["running_costs_BO_ALL"]), axis=0)
    + 459.0
    + 0.5730027548
)
BO_COA_YIELD = np.mean(np.array(RESULTS_ANILINE[i + 1]["y_better_BO_ALL"]), axis=0)
BO_COA_COSTS = (
    np.mean(np.array(RESULTS_ANILINE[i + 1]["running_costs_BO_ALL"]), axis=0)
    + 459.0
    + 0.5730027548
)

ITERATIONS = np.arange(len(BO_YIELD)) + 1


ax2[0, j].plot(
    ITERATIONS,
    BO_YIELD,
    label="BO",
    color="blue",
    marker="o",
    ls="--",
    alpha=0.5,
)

ax2[0, j].plot(
    ITERATIONS,
    BO_COA_YIELD,
    label="BO-COA",
    color="red",
    marker="o",
    ls="--",
    alpha=0.5,
)

ax2[1, j].set_ylabel(r"$\sum \rm cost ~ [\$]$")
ax2[1, j].plot(
    ITERATIONS,
    BO_COSTS,
    label="BO",
    color="blue",
    marker="o",
    ls="--",
    alpha=0.5,
)

ax2[1, j].plot(
    ITERATIONS,
    BO_COA_COSTS,
    label="BO-COI",
    color="red",
    marker="o",
    ls="--",
    alpha=0.5,
)

ax2[0, 0].legend(loc="lower right", fontsize=16, frameon=False)

plt.tight_layout()
plt.savefig("Baumgartner_cheapest.pdf")

pdb.set_trace()

import matplotlib.pyplot as plt
import numpy as np
from utils import *
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import seaborn as sns

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 10
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titlesize"] = 20
# fontsize
plt.rcParams["xtick.labelsize"] = 32
# x and y label size
plt.rcParams["ytick.labelsize"] = 32


def generate_color_scale(iterations, cmap_name="coolwarm"):
    cmap = plt.get_cmap(cmap_name)
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, iterations)]

PLOT_UNCERTAINTY = False

ITERATIONS = np.arange(31) # + 1
RESULTS = load_pkl("results_DirectAryl.pkl")


BO_COI_YIELD, BO_COI_YIELD_STD = np.mean(np.array(RESULTS[0]["y_better_BO_ALL"]), axis=0), np.std(np.array(RESULTS[0]["y_better_BO_ALL"]), axis=0)
BO_COI_COSTS, BO_COI_COSTS_STD = np.mean(np.array(RESULTS[0]["running_costs_BO_ALL"]), axis=0),  np.std(np.array(RESULTS[0]["running_costs_BO_ALL"]), axis=0)
# np.array(RESULTS[0]["running_costs_BO_ALL"])[1]
# np.mean(np.array(RESULTS[0]["running_costs_BO_ALL"]), axis=0)

BO_YIELD, BO_YIELD_STD = np.mean(
    np.array(RESULTS[1]["y_better_BO_ALL"]), axis=0
), np.std(np.array(RESULTS[1]["y_better_BO_ALL"]), axis=0)
BO_COSTS,BO_COSTS_STD  = np.mean(np.array(RESULTS[1]["running_costs_BO_ALL"]), axis=0), np.std(np.array(RESULTS[1]["running_costs_BO_ALL"]), axis=0)
# np.array(RESULTS[1]["running_costs_BO_ALL"])[1]
# np.mean(np.array(RESULTS[1]["running_costs_BO_ALL"]), axis=0)

# pdb.set_trace()
BO_COI_COSTS, BO_COSTS = BO_COI_COSTS + 24.0, BO_COSTS + 24.0

sns.set_context("poster")  # This sets the context to "poster", which is similar to using 'seaborn-poster'
sns.set(style="whitegrid")  # Optionally set a style, "whitegrid" is just an example

fig1, ax1 = plt.subplots(2, 1, figsize=(7, 7))

for i in range(2):  # Row index
    ax = ax1[i]
    ax.spines['top'].set_visible(False)    # Make the top axis line for a plot invisible
    ax.spines['right'].set_visible(False)  # Make the right axis line for a plot invisible
    # Enhance the visibility of axis lines by increasing their linewidth
    ax.spines['bottom'].set_linewidth(4)  # Enhance bottom spine
    ax.spines['left'].set_linewidth(4)    # Enhance left spine
    #spines color black 
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')


ax1[0].plot(
    ITERATIONS,
    BO_COI_YIELD,
    label="CIBO",
    color="#F28E2B",
    marker="o",
    ls="--",
    alpha=0.5,
    ms=10,
)

ax1[1].plot(
    ITERATIONS,
    BO_COI_COSTS,
    label="CIBO",
    color="#F28E2B",
    marker="o",
    ls="--",
    alpha=0.5,
    ms=10,
)

ax1[0].plot(
    ITERATIONS,
    BO_YIELD,
    label="BO",
    color="#4E79A7",
    marker="^",
    ls="--",
    alpha=0.5,
    ms=8,
)
ax1[1].plot(
    ITERATIONS,
    BO_COSTS,
    label="BO",
    color="#4E79A7",
    marker="^",
    ls="--",
    alpha=0.5,
    ms=8,
)


if PLOT_UNCERTAINTY:
    ax1[0].fill_between(
        ITERATIONS,
        BO_COI_YIELD - BO_COI_YIELD_STD,
        BO_COI_YIELD + BO_COI_YIELD_STD,
        alpha=0.2,
        color="red",
    )
    ax1[1].fill_between(
        ITERATIONS,
        BO_COI_COSTS - BO_COI_COSTS_STD,
        BO_COI_COSTS + BO_COI_COSTS_STD,
        alpha=0.2,
        color="red",
    )
    ax1[0].fill_between(
        ITERATIONS,
        BO_YIELD - BO_YIELD_STD,
        BO_YIELD + BO_YIELD_STD,
        alpha=0.2,
        color="#4E79A7",
    )

    ax1[1].fill_between(
        ITERATIONS,
        BO_COSTS - BO_COSTS_STD,
        BO_COSTS + BO_COSTS_STD,
        alpha=0.2,
        color="#4E79A7",
    )

# make x axis ticks integers
ax1[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax1[1].xaxis.set_major_locator(MaxNLocator(integer=True))


ax1[0].set_ylabel("Yield [%]", fontsize=18)
ax1[1].set_ylabel(r"$\sum \rm cost ~ [\$]$", fontsize=18)
ax1[1].set_xlabel("Iteration", fontsize=18)
ax1[0].legend(loc="lower right", fontsize=18)
# Increase tick label sizes
ax1[0].tick_params(axis='both', labelsize=16)  # Adjusts both x and y axis ticks
ax1[1].tick_params(axis='both', labelsize=16)  # Adjusts both x and y axis ticks

# make tight
ax1[0].set_xlim([0, 20])
ax1[1].set_xlim([0, 20])
plt.tight_layout()

plt.savefig("DirectArylation.pdf")
plt.close()


RESULTS = load_pkl("results_Baumgartner_cheapest.pkl")


fig2, ax2 = plt.subplots(2, 4, figsize=(14, 7))
for i in range(2):  # Row index
    for j in range(4):  # Column index
        ax = ax2[i, j]
        ax.spines['top'].set_visible(False)    # Make the top axis line for a plot invisible
        ax.spines['right'].set_visible(False)  # Make the right axis line for a plot invisible
        # Enhance the visibility of axis lines by increasing their linewidth
        ax.spines['bottom'].set_linewidth(4)  # Enhance bottom spine
        ax.spines['left'].set_linewidth(4)    # Enhance left spine
        #spines color black 
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')

nulceophiles = [
    "Benzamide",
    "Phenethylamine",
    "Morpholine",
]
# plt.plot()
plt.show()
# pdb.set_trace()


nucleophiles_legends  = {
    "Benzamide": "CC-Be",
    "Phenethylamine": "CC-Ph",
    "Morpholine": "CC-Mo",
    "Aniline": "CC-An"
}


for i, j in zip([0, 2, 4, 6], [0, 1, 2, 3]):
    nucleophile = RESULTS[i]["settings"]["nucleophile"]
    BO_YIELD = np.mean(np.array(RESULTS[i]["y_better_BO_ALL"]), axis=0)
    BO_COSTS = np.mean(np.array(RESULTS[i]["running_costs_BO_ALL"]), axis=0)

    BO_COA_YIELD = np.mean(np.array(RESULTS[i + 1]["y_better_BO_ALL"]), axis=0)
    BO_COA_COSTS = np.mean(np.array(RESULTS[i + 1]["running_costs_BO_ALL"]), axis=0)
    ITERATIONS = np.arange(len(BO_YIELD))# + 1
    if nucleophile == "Morpholine":
        BO_COSTS += 548 + 2.8
        BO_COA_COSTS += 548 + 2.8
        # pdb.set_trace()
        ax2[1, j].axvline(x=8, color="black", linestyle="--", alpha=1.0)
    else:
        BO_COSTS += 65.2 + 2.8
        BO_COA_COSTS += 65.2 + 2.8

    if nucleophile == "Benzamide":
        ax2[1, j].axvline(x=3, color="black", linestyle="--", alpha=1.0)

    if nucleophile == "Phenethylamine":
        ax2[1, j].axvline(x=3, color="black", linestyle="--", alpha=1.0)

    ax2[0, j].set_ylim([0, 105])
    ax2[0, j].set_xlim([0, len(ITERATIONS) + 1])    
    ax2[1, j].set_ylim([0, 2600])

    ax2[0, j].set_title(f"{nucleophile}", fontweight="bold")
    ax2[0, j].set_title(f"{nucleophiles_legends[nucleophile]}", fontweight="bold", fontsize=18)

    if nucleophile != "Aniline":
        ax2[0, j].plot(
            ITERATIONS,
            BO_YIELD,
            label="BO",
            color="#4E79A7",
            marker="^",
            ls="--",
            alpha=0.5,
            ms=8,
        )

        ax2[0, j].plot(
            ITERATIONS,
            BO_COA_YIELD,
            label="CIBO",
            color="#F28E2B",
            marker="o",
            ls="--",
            alpha=0.5,
            ms=10,
        )
        # plot horizontal grey dotted line at 70 % yield
        ax2[0, j].axhline(y=70, color="black", linestyle="--", alpha=1.0)
        # ax2[1, j].set_ylabel(r"$\sum \rm cost ~ [\$]$")
        ax2[1, j].plot(
            ITERATIONS,
            BO_COSTS,
            label="BO",
            color="#4E79A7",
            marker="^",
            ls="--",
            alpha=0.5,
            ms=8,
        )

        ax2[1, j].plot(
            ITERATIONS,
            BO_COA_COSTS,
            label="CIBO",
            color="#F28E2B",
            marker="o",
            ls="--",
            alpha=0.5,
            ms=10,
        )
    for k in range(2):
        ax2[k, j].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2[k, j].set_xlim([0, len(ITERATIONS) + 1])
        # ticsk
        ax2[k, j].set_xticks(np.arange(0, len(ITERATIONS) + 1, 5))
        ax2[k, j].tick_params(axis="both", which="major", labelsize=18)

ax2[0][0].set_ylabel("Yield [%]", fontsize=18)
ax2[0, 0].legend(loc="lower right", fontsize=18, frameon=False)
ax2[1, 0].set_xlabel("Iteration", fontsize=18)
ax2[1, 1].set_xlabel("Iteration", fontsize=18)
ax2[1, 2].set_xlabel("Iteration", fontsize=18)
ax2[1, 3].set_xlabel("Iteration", fontsize=18)


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
# pdb.set_trace()
ITERATIONS = np.arange(len(BO_YIELD)) + 1


ax2[0, j].plot(
    ITERATIONS,
    BO_YIELD,
    label="BO",
    color="#4E79A7",
    marker="^",
    ls="--",
    alpha=0.5,
    ms=8,
)

ax2[0, j].plot(
    ITERATIONS,
    BO_COA_YIELD,
    label="CIBO",
    color="#F28E2B",
    marker="o",
    ls="--",
    alpha=0.5,
    ms=10,
)

ax2[1, j].set_ylabel(r"$\sum \rm cost ~ [\$]$", fontsize=18)
ax2[1, j].plot(
    ITERATIONS,
    BO_COSTS,
    label="BO",
    color="#4E79A7",
    marker="^",
    ls="--",
    alpha=0.5,
    ms=8,
)

ax2[1, j].plot(
    ITERATIONS,
    BO_COA_COSTS,
    label="CIBO",
    color="#F28E2B",
    marker="o",
    ls="--",
    alpha=0.5,
    ms=10,
)

ax2[0, 0].legend(loc="lower right", fontsize=16, frameon=False)
ax2[0, 0].axhline(y=70, color="black", linestyle="--", alpha=1.0)
# vertical line at 20 iterations
ax2[1, 0].axvline(x=4, color="black", linestyle="--", alpha=1.0)
plt.tight_layout()
plt.savefig("Baumgartner_cheapest.pdf")

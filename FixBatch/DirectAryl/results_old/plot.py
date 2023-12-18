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

RESULTS_RANDOM = load_pkl("results_random.pkl")

random_results = np.mean(np.array(RESULTS_RANDOM[0]["y_better_RANDOM_ALL"]), axis=0)
random_costs = np.mean(np.array(RESULTS_RANDOM[0]["running_costs_RANDOM_ALL"]), axis=0)

bo_adaptive_2_results_20 = np.mean(np.array(RESULTS[0]["y_better_BO_ALL"]), axis=0)
bo_adaptive_2_costs_20 = np.mean(np.array(RESULTS[0]["running_costs_BO_ALL"]), axis=0)


bo_results_aware_constant_100 = np.mean(np.array(RESULTS[1]["y_better_BO_ALL"]), axis=0)
bo_costs_aware_constant_100 = np.mean(
    np.array(RESULTS[1]["running_costs_BO_ALL"]), axis=0
)

bo_results_aware_increasing_100 = np.mean(
    np.array(RESULTS[2]["y_better_BO_ALL"]), axis=0
)
bo_costs_aware_increasing_100 = np.mean(
    np.array(RESULTS[2]["running_costs_BO_ALL"]), axis=0
)


bo_results_aware_decreasing_1000 = np.mean(
    np.array(RESULTS[3]["y_better_BO_ALL"]), axis=0
)
bo_costs_aware_decreasing_1000 = np.mean(
    np.array(RESULTS[3]["running_costs_BO_ALL"]), axis=0
)


bo_results_aware_constant_150 = np.mean(np.array(RESULTS[4]["y_better_BO_ALL"]), axis=0)
bo_costs_aware_constant_150 = np.mean(
    np.array(RESULTS[4]["running_costs_BO_ALL"]), axis=0
)


bo_adaptive_results_100 = np.mean(np.array(RESULTS[5]["y_better_BO_ALL"]), axis=0)
bo_adaptive_costs_100 = np.mean(np.array(RESULTS[5]["running_costs_BO_ALL"]), axis=0)


bo_adaptive_results_50 = np.mean(np.array(RESULTS[6]["y_better_BO_ALL"]), axis=0)
bo_adaptive_costs_50 = np.mean(np.array(RESULTS[6]["running_costs_BO_ALL"]), axis=0)

bo_adaptive_results_150 = np.mean(np.array(RESULTS[7]["y_better_BO_ALL"]), axis=0)
bo_adaptive_costs_150 = np.mean(np.array(RESULTS[7]["running_costs_BO_ALL"]), axis=0)


bo_results_unaware = np.mean(np.array(RESULTS[8]["y_better_BO_ALL"]), axis=0)
bo_costs_unaware = np.mean(np.array(RESULTS[8]["running_costs_BO_ALL"]), axis=0)

plt.style.use("seaborn-poster")  # Apply a global aesthetic style.
ITERATIONS_1 = np.arange(len(random_costs)) + 1
ITERATIONS_2 = np.arange(len(bo_adaptive_results_150)) + 1
ITERATIONS_3 = np.arange(len(bo_adaptive_2_results_20)) + 1

fig1, ax1 = plt.subplots(2, 1, figsize=(7, 7))

ax1[0].plot(
    ITERATIONS_1, bo_results_unaware, label="BO", color="red", marker="o", ls="--"
)
ax1[0].plot(
    ITERATIONS_1,
    bo_results_aware_increasing_100,
    label="BO-COST-INC-100",
    color="navy",
    marker="o",
    ls="-.",
)

ax1[0].plot(
    ITERATIONS_2,
    bo_adaptive_results_50,
    label="BO-ADAPT-50",
    color="brown",
    marker="o",
    ls="-.",
)


ax1[0].plot(
    ITERATIONS_2,
    bo_adaptive_results_150,
    label="BO-ADAPT-150",
    color="yellow",
    marker="o",
    ls="-.",
)

ax1[0].plot(ITERATIONS, random_results, label="RS", color="green", marker="o", ls="--")
ax1[0].plot(
    ITERATIONS_3,
    bo_adaptive_2_results_20,
    label="BO-ADAPT-2",
    marker="o",
    color="black",
)


ax1[1].plot(ITERATIONS, bo_costs_unaware, label="BO", color="red", marker="o", ls="--")
ax1[1].plot(
    ITERATIONS_1,
    bo_costs_aware_increasing_100,
    label="BO-COST-INC-100",
    color="navy",
    marker="o",
    ls="-.",
)

ax1[1].plot(
    ITERATIONS_2,
    bo_adaptive_costs_150,
    label="BO-ADAPT-150",
    color="yellow",
    marker="o",
    ls="-.",
)

ax1[1].plot(
    ITERATIONS_2,
    bo_adaptive_costs_50,
    label="BO-ADAPT-50",
    color="brown",
    marker="o",
    ls="-.",
)


ax1[1].plot(ITERATIONS, random_costs, label="RS", color="green", marker="o", ls="--")
ax1[1].plot(
    ITERATIONS_3, bo_adaptive_2_costs_20, label="BO-ADAPT-2", marker="o", color="black"
)


# make x axis ticks integers
ax1[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax1[1].xaxis.set_major_locator(MaxNLocator(integer=True))


ax1[0].set_ylabel("Yield [%]")  # Assuming yield is in percentage.
ax1[1].set_ylabel("Sum(Cost) [$]")
ax1[1].set_xlabel("Iteration")
ax1[0].legend(loc="lower right", fontsize=13)
# ax1[1].legend(loc="lower right", fontsize=13)

# make tight
ax1[0].set_xlim([1, 45])
ax1[1].set_xlim([1, 45])
plt.tight_layout()

plt.savefig("optimization.png")
plt.close()


fig2, ax2 = plt.subplots(1, 1, figsize=(7, 7))


ax2.plot(
    random_costs, random_results, label="RS", color="black", marker="o", ls="-"
)

ax2.plot(
    bo_adaptive_2_costs_20, bo_adaptive_2_results_20, label="BO-ADAPT_2_20", color="red"
)

ax2.plot(
    bo_costs_aware_constant_100,
    bo_results_aware_constant_100,
    label="BO-COST-CONST-100",
    color="navy",
)

ax2.plot(
    bo_costs_aware_increasing_100,
    bo_results_aware_increasing_100,
    label="BO-COST-INC-100",
    color="brown",
    marker="o",
    ls="-.",
)

ax2.plot(
    bo_costs_aware_decreasing_1000,
    bo_results_aware_decreasing_1000,
    label="BO-COST-DESC-1000",
    color="green",
    marker="o",
    ls="-.",
)

ax2.plot(
    bo_costs_aware_constant_150,
    bo_results_aware_constant_150,
    label="BO-COST-CONST-150",
    color="orange",
    marker="o",
    ls="-.",
)

ax2.plot(
    bo_adaptive_costs_100,
    bo_adaptive_results_100,
    label="BO-ADAPT-100",
    color="yellow",
    marker="o",
    ls="-.",
)

ax2.plot(
    bo_adaptive_costs_50,
    bo_adaptive_results_50,
    label="BO-ADAPT-50",
    color="purple",
    marker="o",
    ls="-.",
)

ax2.plot(
    bo_adaptive_costs_150,
    bo_adaptive_results_150,
    label="BO-ADAPT-150",
    color="pink",
    marker="o",
    ls="-.",
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
plt.close()


fig3, ax3 = plt.subplots(1, 1, figsize=(7, 7))


ax3.plot(
    ITERATIONS_1,
    (random_results / (random_costs + 1)),
    label="RS",
    color="black",
    marker="o",
    ls="-",
)

ax3.plot(
    ITERATIONS_3,
    (bo_adaptive_2_results_20 / (bo_adaptive_2_costs_20 + 1)),
    label="BO-ADAPT_2_20",
    color="red",
)

ax3.plot(
    ITERATIONS_1,
    (bo_results_aware_constant_100 / (bo_costs_aware_constant_100 + 1)),
    label="BO-COST-CONST-100",
    color="navy",
)

ax3.plot(
    ITERATIONS_1,
    (bo_results_aware_increasing_100 / (bo_costs_aware_increasing_100 + 1)),
    label="BO-COST-INC-100",
    color="brown",
    marker="o",
    ls="-.",
)

ax3.plot(
    ITERATIONS_1,
    (bo_results_aware_decreasing_1000 / (bo_costs_aware_decreasing_1000 + 1)),
    label="BO-COST-DESC-1000",
    color="green",
    marker="o",
    ls="-.",
)

ax3.plot(
    ITERATIONS_1,
    (bo_results_aware_constant_150 / (bo_costs_aware_constant_150 + 1)),
    label="BO-COST-CONST-150",
    color="orange",
    marker="o",
    ls="-.",
)

ax3.plot(
    ITERATIONS_2,
    (bo_adaptive_results_100 / (bo_adaptive_costs_100 + 1)),
    label="BO-ADAPT-100",
    color="yellow",
    marker="o",
    ls="-.",
)

ax3.plot(
    ITERATIONS_2,
    (bo_adaptive_results_50 / (bo_adaptive_costs_50 + 1)),
    label="BO-ADAPT-50",
    color="purple",
    marker="o",
    ls="-.",
)

ax3.plot(
    ITERATIONS_2,
    (bo_adaptive_results_150 / (bo_adaptive_costs_150 + 1)),
    label="BO-ADAPT-150",
    color="pink",
    marker="o",
    ls="-.",
)

ax3.plot(
    ITERATIONS_1,
    (bo_results_unaware / (bo_costs_unaware + 1)),
    label="BO",
    color="red",
    marker="o",
    ls="--",
)







ax3.set_xlabel("Iteration")
ax3.set_ylabel("Yield [%]/Sum(Cost) [$]")  # Assuming yield is in percentage.
ax3.legend(loc="upper right", fontsize=13)
ax3.set_yscale("log")
ax3.set_xscale("log")
# make tight
plt.tight_layout()
plt.savefig("comparison2.png")
plt.close()

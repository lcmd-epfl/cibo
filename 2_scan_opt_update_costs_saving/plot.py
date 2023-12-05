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


bo_results_aware_constant_20 = np.mean(np.array(RESULTS[0]["y_better_BO_ALL"]), axis=0)
bo_costs_aware_constant_20 = np.mean(np.array(RESULTS[0]["running_costs_BO_ALL"]), axis=0)

bo_results_aware_increasing_20 = np.mean(np.array(RESULTS[1]["y_better_BO_ALL"]), axis=0)
bo_costs_aware_increasing_20 = np.mean(np.array(RESULTS[1]["running_costs_BO_ALL"]), axis=0)



bo_results_aware_decreasing_600 = np.mean(np.array(RESULTS[2]["y_better_BO_ALL"]), axis=0)
bo_costs_aware_decreasing_600 = np.mean(np.array(RESULTS[2]["running_costs_BO_ALL"]), axis=0)



bo_results_aware_constant_30 = np.mean(np.array(RESULTS[3]["y_better_BO_ALL"]), axis=0)
bo_costs_aware_constant_30 = np.mean(np.array(RESULTS[3]["running_costs_BO_ALL"]), axis=0)



bo_results_unaware = np.mean(np.array(RESULTS[4]["y_better_BO_ALL"]), axis=0)
bo_costs_unaware = np.mean(np.array(RESULTS[4]["running_costs_BO_ALL"]), axis=0)

plt.style.use("seaborn-poster")  # Apply a global aesthetic style.

fig1, ax1 = plt.subplots(2, 1, figsize=(7, 7))

ax1[0].plot(
    ITERATIONS, bo_results_unaware, label="BO", color="red", marker="o", ls="--"
)
ax1[0].plot(
    ITERATIONS, bo_results_aware_increasing_20, label="BO-COST", color="navy", marker="o", ls="-."
)

ax1[1].plot(ITERATIONS, bo_costs_unaware, label="BO", color="red", marker="o", ls="--")
ax1[1].plot(
    ITERATIONS, bo_costs_aware_increasing_20, label="BO-COST", color="navy", marker="o", ls="-."
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
    bo_costs_unaware, bo_results_unaware, label="BO", color="red", marker="o", ls="--"
)

ax2.plot(
    bo_costs_aware_increasing_20, bo_results_aware_increasing_20, label="BO-COST-INC-20", color="navy", marker="o", ls="-."
)









ax2.plot(
    bo_costs_aware_constant_20, bo_results_aware_constant_20, label="BO-COST-CONST-20", color="orange", marker="o", ls="-."
)


ax2.plot(
    bo_costs_aware_decreasing_600, bo_results_aware_decreasing_600, label="BO-COST-DESC-600", color="green", marker="o", ls="-."
)


ax2.plot(
    bo_costs_aware_constant_30, bo_results_aware_constant_30, label="BO-COST-CONST-30", color="yellow", marker="o", ls="-."
)


ax2.set_xlabel("Sum(Cost) [$]")
ax2.set_ylabel("Yield [%]")  # Assuming yield is in percentage.
ax2.legend(loc="lower right", fontsize=13)
# make tight
plt.tight_layout()
plt.savefig("comparison.png")
plt.close()
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import pdb
import matplotlib.colors as mcolors


def generate_color_scale(iterations, cmap_name='coolwarm'):
    cmap = plt.get_cmap(cmap_name)
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, iterations)]


REACTTION_1 = "ebdo_direct_arylation"
REACTTION_2 = "buchwald"
RESULTS = load_pkl("results.pkl")

ITERATIONS = [1,5, 10, 15, 20]

diff_1 = []
diff_2 = []
for i in range(12):
    curr_diff = []
    for it in ITERATIONS:
        random = np.mean(np.array(RESULTS[i]["y_better_RANDOM_ALL"]), axis=0)[it]
        bo     = np.mean(np.array(RESULTS[i]["y_better_BO_ALL"]), axis=0)[it]


        max_n_BO            = reaching_max_n(RESULTS[i]["y_better_BO_ALL"])
        max_n_RANDOM        = reaching_max_n(RESULTS[i]["y_better_RANDOM_ALL"])

        #diff = bo_best/max_n_BO - random_best/max_n_RANDOM
        diff = bo - random
        print(i, diff, max_n_BO, max_n_RANDOM, RESULTS[i]["settings"]["dataset"])

        curr_diff.append(diff)

    if i < 6:
        diff_1.append(curr_diff)
    else:
        diff_2.append(curr_diff)




diff_1, diff_2 = np.array(diff_1), np.array(diff_2)

#create two plot for each reaction

fig, ax = plt.subplots()


#create an iterable colorscale in form of a list
colors = generate_color_scale(len(ITERATIONS))

ax.set_title("Difference in utility between BO and RS for {}".format(REACTTION_1))
ax.set_xlabel("Max batch cost")
ax.set_ylabel("Utility difference")
for ind, it in enumerate(ITERATIONS):
    print(ind, it)
    ax.plot(np.arange(6), diff_1[:,ind],c=colors[ind], label="Iteration {}".format(it))

ax.set_xticks(np.arange(6))
ax.set_xticklabels(np.arange(6))
#ax.set_yscale("log")
ax.legend()

fig.savefig("./figures/diff_{}.png".format(REACTTION_1))

fig, ax = plt.subplots()
ax.set_title("Difference in utility between BO and RS for {}".format(REACTTION_2))
ax.set_xlabel("Max batch cost")
ax.set_ylabel("Utility difference")
for ind, it in enumerate(ITERATIONS):
    ax.plot(np.arange(6), diff_2[:,ind], c=colors[ind],label="Iteration {}".format(it))

ax.set_xticks(np.arange(6))
ax.set_xticklabels(np.arange(6))
#ax.set_yscale("log")
ax.legend()

fig.savefig("./figures/diff_{}.png".format(REACTTION_2))


#pdb.set_trace()

#ebdo_direct_arylation
#buchwald
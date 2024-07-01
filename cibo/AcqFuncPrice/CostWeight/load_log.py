import pdb
from utils import *
import pandas as pd

log = load_pkl("results.pkl")
exp = np.array(log[1]["exp_log"])


best_vals = [log[1]["y_better_BO_ALL"][0][0]]

for i in range(len(exp)):

    batch_max = exp[i][:, -1].max()
    if batch_max > best_vals[-1]:
        best_vals.append(batch_max)
    else:
        best_vals.append(best_vals[-1])

best_vals = np.array(best_vals)
plt.plot(best_vals)
plt.savefig("best_vals.png")

reshaped_data = exp.reshape(-1, exp.shape[2])
reshaped_data = exp.reshape(-1, exp.shape[2])

# Create a DataFrame
df = pd.DataFrame(
    reshaped_data, columns=["Base", "Ligand", "Solvent", "c", "T", "Yield"]
)

# Add iteration and batchsize columns
df["Iteration"] = np.repeat(np.arange(exp.shape[0]), exp.shape[1])
df["Batchsize"] = 5  # constant batchsize as per the user's data structure

#save pd as csv
df.to_csv("exp_log.csv")
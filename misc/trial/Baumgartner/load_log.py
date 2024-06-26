import pdb
from utils import *
import pandas as pd

log = load_pkl("results.pkl")

for i in range(len(log)):
    exp = np.array(log[i]["exp_log"])
    # pdb.set_trace()
    nucleophile = log[i]["settings"]["nucleophile"]

    cost = log[i]["settings"]["cost_aware"]

    best_vals = [log[i]["y_better_BO_ALL"][0][0]]

    for i in range(len(exp)):
        batch_max = exp[i][:, -1].max()
        if batch_max > best_vals[-1]:
            best_vals.append(batch_max)
        else:
            best_vals.append(best_vals[-1])

    best_vals = np.array(best_vals)
    fig, ax = plt.subplots()
    ax.plot(best_vals)
    plt.savefig("./log_files/best_vals_{}_{}.png".format(nucleophile, cost))
    plt.close()

    # Reshape the array to a 2D array where each row represents a batch in an iteration
    reshaped_data = exp.reshape(-1, exp.shape[2])

    # Create a DataFrame with the appropriate column names
    df = pd.DataFrame(
        reshaped_data,
        columns=[
            "Precatalyst",
            "Solvent",
            "Base",
            "Base_Conc",
            "T",
            "Base_Eq",
            "Residence_Time",
            "Yield",
        ],
    )

    # Add iteration and batchsize columns
    df["Iteration"] = np.repeat(np.arange(exp.shape[0]), exp.shape[1])
    df["Batchsize"] = 5  # constant batchsize as per the user's data structure

    # Reorder the columns as per user's request
    df = df[
        [
            "Iteration",
            "Batchsize",
            "Precatalyst",
            "Solvent",
            "Base",
            "Base_Conc",
            "T",
            "Base_Eq",
            "Residence_Time",
            "Yield",
        ]
    ]

    # Code to export the DataFrame to a CSV file
    df.to_csv(f"./log_files/{nucleophile}_{cost}.csv", index=False)

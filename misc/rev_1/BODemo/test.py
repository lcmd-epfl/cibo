import pickle
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Replace 'your_file.pkl' with the path to your pickle file
file_path = 'BO_data_normal.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

#plt.plot(data[:,0][:100], data[:,1][:100], "r")
#instead disply density in 2 d 
    
df = {'x': data[:,0], 'y': data[:,1]}
sns.kdeplot(x=df['x'], y=df['y'], shade=True, cmap="Reds", shade_lowest=False)
plt.savefig("test.png")


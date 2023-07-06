
import matplotlib.pyplot as plt
import numpy as np

from dysts.flows import *

import json
from scipy.integrate import solve_ivp


filename_old = "d:/2022-23/Okul/Dersler/BS723/[05][03] Experimentation/moreSystems/chaotic_attractors_old.json"
filename = "d:/2022-23/Okul/Dersler/BS723/[05][03] Experimentation/moreSystems/chaotic_attractors.json"

# read file contents
with open(filename_old, 'r') as read_file:
    data = json.load(read_file)

print(len(data.keys()))

# dims_hist = []

# for key in data.keys():
#     # append dimensions
#     dims_hist.append(data[key]["dimension"])
#     print(data[key]["dimension"])

# plt.hist(dims_hist,bins=np.arange(min(dims_hist), max(dims_hist) +0.75, 0.75), color='teal')
# plt.xlim(max(dims_hist)+1, min(dims_hist)-1)
# plt.show()


# print the file contents
for key in data.keys():
    scale = []
    offset = []

    model = eval(key+"()")
    sol = model.make_trajectory(1000, resample=True)
    
    for dimension in  range(len(data[key]["initial_conditions"])):
        scale.append( np.max(sol[:,dimension])-np.min(sol[:,dimension]))
        offset.append( (np.max(sol[:,dimension])+np.min(sol[:,dimension]))/2)

    data[key]["dimension"] = len(data[key]["initial_conditions"])
    data[key]["offset"] = offset
    data[key]["scale"] = scale

# write changes to file in json format
with open(filename, 'w') as write_file:
    json.dump(data, write_file, indent=4)


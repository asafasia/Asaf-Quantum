from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from helper import plotter
from helper import experiment_results
from helper.utility_functions import correct_axis, cos_wave_exp

# %% choose data
date = '17_12_2023'

exp_name = 'power_broadening'

n_index = 6

data = experiment_results.ExperimentData.load_data(date, exp_name, n_index)
print(data['meta_data'].keys())
print(data['meta_data']["experiment_properties"].keys())


qubit = data['meta_data']["experiment_properties"]['qubit']
qubit_parameters = data['meta_data']["experiment_properties"]['qubit_parameters']

dfs = np.array(data['data']['x_data'])
dAs = np.array(data['data']['y_data'])
z = np.array(data['data']['z_data'])

# %% colormesh

dfs_mesh, dA_mesh, = np.meshgrid(dfs, dAs)

print(z.shape)

plt.title(f"Power Broadening {qubit} ")
plt.pcolormesh((dfs_mesh - qubit_parameters[qubit]['qb_freq']) * 1e-6, dA_mesh * 1e-6, z)
plt.show()


###### properties ######
for key in data['meta_data']["experiment_properties"].keys():
    print(key, data['meta_data']["experiment_properties"][key])
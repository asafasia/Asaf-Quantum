# written by Guy. 25.12.23
# example script showing how to use labber_util.create_logfile() to put data from an experiment into labber
# using the data from the latest spin-locking experiment

import json
import numpy as np
import labber_util as lu

# load data from json file
file = open(r"../results/data/22_12_2023/spin_locking_long_runnnnnnnn_1.json")
result = json.load(file)

# unpack result to a format similar to what we have in the end of an experiment script
result_X = np.array(result["data"]["x_data_r"]) + 1j * np.array(result["data"]["x_data_c"])
result_Y = np.array(result["data"]["y_data_r"]) + 1j * np.array(result["data"]["y_data_c"])
result_Z = np.array(result["data"]["z_data_r"]) + 1j * np.array(result["data"]["z_data_c"])

delay_vec = result["data"]["t"]
detuning_vec = result["data"]["f"]

meta_data = result["meta_data"]

# add tags and user
meta_data["tags"] = ["Nadav-Lab", "spin-locking", "overnight"]
meta_data["user"] = "Guy"

# arrange data in a form that is more suitable for labber (separate sweep parameters from measured ones, include units
# etc.)
measured_data = dict(X=result_X, Y=result_Y, Z=result_Z)
sweep_parameters = dict(hold_time=delay_vec, detuning=detuning_vec)
units = dict(hold_time="s", detuning="Hz")

exp_result = dict(measured_data=measured_data, sweep_parameters=sweep_parameters, units=units, meta_data=meta_data)

# create logfile
lu.create_logfile("spin_locking", **exp_result, loop_type="2d")


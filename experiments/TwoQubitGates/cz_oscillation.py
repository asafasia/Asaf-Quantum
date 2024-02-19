# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:45:13 2024

@author: stud
"""

# %% Imports

# LabOne Q:
from laboneq.simple import *
# Helpers:
from laboneq.contrib.example_helpers.plotting.plot_helpers import (
    plot_results,
    plot_simulation,
)
from laboneq.contrib.example_helpers.generate_example_datastore import (
    generate_example_datastore,
    get_first_named_entry,
)
from pathlib import Path
import time
import numpy as np

import matplotlib.pyplot as plt
import math

from helper.exp_helper import *
from helper.pulses import *
from qubit_parameters import qubit_parameters, update_qp
from helper.kernels import kernels

from qubit_parameters import qubit_parameters

import json
from datetime import datetime
import os

from scipy.optimize import curve_fit

# %% devise setup


coupler = 'c43'

if coupler == 'c13':
    qubit_m = "q1"
    qubit_s = "q3"
    n_port = 5

elif coupler == 'c23':
    qubit_m = "q3"
    qubit_s = "q2"
    n_port = 6
elif coupler == 'c43':
    qubit_s = "q3"
    qubit_m = "q4"
    n_port = 7

mode = 'disc'
modulation_type = 'hardware' if mode == 'spec' else 'software'
if mode == 'spec':
    acquisition_type = AcquisitionType.SPECTROSCOPY
elif mode == 'int':
    acquisition_type = AcquisitionType.INTEGRATION
elif mode == 'disc':
    acquisition_type = AcquisitionType.DISCRIMINATION

kernel = readout_pulse(qubit_m) if mode == 'spec' else kernels[qubit_s]

exp = initialize_exp()
device_setup = exp.create_device_setup(modulation_type)
exp_signals = exp.signals(qubit_s)
signal_map_default = exp.signal_map_default(qubit_s)
# %%
session = Session(device_setup=device_setup)
session.connect(do_emulation=False)

# %% parameters for user

# central frequency for drive frequency sweep:
central_frequency = qubit_parameters[qubit_s]["qb_freq"]

# True for pulse simulation plots
simulate = False
plot_from_json = False
# %%parameters for sweeping

num_averages = 1000

flux = -0.2

max_time = 0.5e-6  # [sec]
time_step_num = 200

# %% parameters for experiment


time_sweep = LinearSweepParameter(
    "freq_sweep", 0, max_time, time_step_num)


# %%
def qubit_spectroscopy(time_sweep):
    # Create qubit spectroscopy Experiment - uses qubit drive, readout drive and data acquisition lines
    exp_qspec = Experiment(
        uid="Qubit Spectroscopy",
        signals=exp_signals,
    )

    with exp_qspec.acquire_loop_rt(
            uid="freq_shots",
            count=num_averages,
            acquisition_type=acquisition_type,
            averaging_mode=AveragingMode.CYCLIC

    ):
        # sweep qubit drive

        with exp_qspec.sweep(uid="qfreq_sweep", parameter=time_sweep):
            with exp_qspec.section(uid="qubit_excitation"):
                exp_qspec.play(signal=f"drive_{qubit_s}", pulse=pulse_library.const(
                    uid=f"pi_pulse_1",
                    length=qubit_parameters[qubit_s]["pi_len"],
                    amplitude=qubit_parameters[qubit_s]["pi_amp"] * 1 / 2,
                    can_compress=False
                ))
            # exp_qspec.play(signal=f"drive_{qubit_m}", pulse=pi_pulse(qubit_m), amplitude=1/2)

            with exp_qspec.section(uid="flux_section", play_after='qubit_excitation'):
                exp_qspec.play(
                    signal=f"flux_{coupler}",
                    pulse=flux_pulse(coupler),
                    amplitude=flux,
                    length=time_sweep)

            with exp_qspec.section(uid="time_delay", play_after="flux_section"):
                exp_qspec.play(signal=f"drive_{qubit_s}", pulse=pi_pulse(qubit_s), amplitude=1 / 2,
                               phase=-2 * np.pi * 0 * time_sweep)
                # exp_qspec.delay(signal="measure", time=10e-9)

            with exp_qspec.section(uid="readout_section", play_after="time_delay"):
                exp_qspec.play(signal="measure", pulse=readout_pulse(qubit_s))

                exp_qspec.acquire(
                    signal="acquire",
                    handle="qubit_spec",
                    kernel=kernel
                )
            # delay between consecutive experiment
            with exp_qspec.section(uid="delay"):
                # relax time after readout - for qubit relaxation to groundstate and signal processing
                exp_qspec.delay(signal="measure", time=120e-6)
    return exp_qspec


# %% Run Exp
# define experiment with frequency sweep for qubit 0
exp_qspec = qubit_spectroscopy(time_sweep)

# apply calibration and signal map for qubit 0
exp_qspec.set_signal_map(signal_map_default)
# %% compile exp
# compile the experiment on the open instrument session


if simulate:
    compiled_qspec = session.compile(exp_qspec)
    plot_simulation(compiled_qspec, start_time=0, length=300e-6)

# %% run the compiled experiemnt
qspec_results = session.run(exp_qspec)
acquire_results = qspec_results.get_data("qubit_spec")

if plot_from_json:

    file_path = r'C:\Users\stud\Documents\GitHub\qhipu-files\LabOne Q\Exp_results\2023-12-20_results\qubit_spectroscopy_q1_13-20-35.json'
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    amplitude = data["plot_vectors"]["amplitude"]
    time_sweep = data["plot_vectors"]["time_sweep"]


else:
    amplitude = np.abs(acquire_results)

# %%

if not plot_from_json:

    # plot vectors

    plot_vectors = {
        'amplitude': amplitude.tolist(),
        'time_sweep': time_sweep.values.tolist(),
    }

    # Additional experiment parameters
    experiment_parameters = {

        'n_avg': num_averages,
        'time_step_num': time_step_num,
    }

    # Add experiment parameters to the data dictionary

    # New parameter: experiment string
    experiment_string = """

    return exp
    """

    data = {
        'experiment_parameters': experiment_parameters,
        'plot_vectors': plot_vectors,
        'qubit_parameters': qubit_parameters,
    }

    data['experiment'] = experiment_string.strip()

    current_date = datetime.now().strftime("%Y-%m-%d")
    # Create a folder based on the current date if it doesn't exist
    folder_path = os.path.join("C:/Users/stud/Documents/GitHub/qhipu-files/LabOne Q/Exp_results",
                               f"{current_date}_results")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Generate a unique filename based on the current time
    current_time = datetime.now().strftime("%H-%M-%S")
    filename = os.path.join(folder_path, f"Cz_oscillation_{qubit_s}_{current_time}.json")

    with open(filename, 'w') as json_file:
        json.dump(data, json_file)  # Adding indent for better readability

    print(f"Data saved to {filename}")

# %% plot slice


save_func(session, "coupler_spec")

# amplitude = correct_axis(amplitude,qubit_parameters[qubit_s]["ge"])
if plot_from_json == False:
    plt.plot(time_sweep, amplitude)
else:
    plt.plot(time_sweep.values, amplitude)


def cos_wave(x, amplitude, T, offset):
    return amplitude * np.cos(2 * np.pi / T * x) + offset


# Assuming your signal is e^(i * theta)
phase_guess = np.angle(np.mean(amplitude))  # Initial guess for the angle

guess = [(max(amplitude) - min(amplitude)) * 10, 1e-7, np.mean(amplitude)]

params, params_covariance = curve_fit(cos_wave, time_sweep, amplitude, p0=guess)

# Plot the amplitude in the first subplot
plt.plot(time_sweep, cos_wave(time_sweep, *params),
         label='Fitted function', color='red')
plt.xlabel('Drive Amplitude [a.u.]')
plt.ylabel('Amplitude [a.u.]')
plt.legend()
# Set a single big title for the entire figure
plt.title(f'Phase Oscillations {qubit_s}', fontsize=18)

phase_shift = 2 * np.pi / params[1] * time_sweep
phase_shift_parameter = params[1]
print("phase_shift_helper =", params[1])

# # Adjust layout and display the plot
# text = f"Phase Shift: {phase_shift:.4f}"
# plt.text(0.65, 0.2, text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8))


plt.xlabel('time [us]')
plt.ylabel('amp [us]')

plt.title(f'Cz Oscillations vs. Coupler Flux {qubit_s} {coupler}', fontsize=18)

plt.tight_layout()
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:05:05 2023

@author: stud
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:22:41 2023

@author: stud
"""

# -*- coding: utf-8 -*-
"""
Amplitude/Power Rabi Created on Thu Nov  2 14:55:13 2023

@author: stud
"""

from laboneq.simple import *
from laboneq.analysis.fitting import oscillatory
from laboneq.contrib.example_helpers.plotting.plot_helpers import (
    plot_results,
    plot_simulation,
)

from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from helper.kernels import kernels
from helper.exp_helper import *
from helper.pulses import *
from pulses import *
from qubit_parameters import *
import labber.labber_util as lu
# %% parameters
qubit = "q5"

mode = 'disc'
modulation_type = 'hardware' if mode == 'spec' else 'software'
if mode == 'spec':
    acquisition_type = AcquisitionType.SPECTROSCOPY
elif mode == 'int':
    acquisition_type = AcquisitionType.INTEGRATION
elif mode == 'disc':
    acquisition_type = AcquisitionType.DISCRIMINATION

kernel = readout_pulse(qubit) if mode == 'spec' else kernels[qubit]

exp = initialize_exp()
device_setup = exp.create_device_setup(modulation_type)
signal_map_default = exp.signal_map_default(qubit)
exp_signals = exp.signals(qubit)

# %%
session = Session(device_setup=device_setup)
session.connect(do_emulation=False)

# %% parameters
simulate = False

exp_repetitions = 250

w = 1e6
sweep_start = 0
sweep_stop = 20e-6
step_num = 200

sweep_time = LinearSweepParameter(uid="sweep", start=sweep_start, stop=sweep_stop, count=step_num)


# %% Experiment Definition


def ramsey_exp(basis):
    exp_rabi = Experiment(
        uid="Qubit Spectroscopy",
        signals=exp_signals,

    )
    #

    with exp_rabi.acquire_loop_rt(
            uid="rabi_shots",
            count=exp_repetitions,
            acquisition_type=acquisition_type,
    ):
        with exp_rabi.sweep(uid="rabi_sweep", parameter=sweep_time):
            with exp_rabi.section(
                    uid="qubit_excitation"
            ):
                exp_rabi.play(
                    signal=f"drive_{qubit}", pulse=pi_pulse(qubit), phase = np.pi/2, amplitude=1 / 2)

                exp_rabi.delay(signal=f"drive_{qubit}", time=sweep_time)

                # exp_rabi.play(
                #     signal=f"drive_{qubit}", pulse=pi_pulse(qubit), phase=2 * np.pi * w * sweep_time, amplitude=1 / 2
                # )

                exp_rabi.play(
                    signal=f"drive_{qubit}", pulse=None, increment_oscillator_phase=2 * np.pi * w * sweep_time)

            with exp_rabi.section(uid="change_of_basis", play_after="qubit_excitation"):
                if basis == 'X':
                    exp_rabi.play(
                        signal=f"drive_{qubit}", pulse=pi_pulse(qubit), phase= np.pi/2, amplitude=1 / 2
                    )
                elif basis == 'Y':
                    exp_rabi.play(
                        signal=f"drive_{qubit}", pulse=pi_pulse(qubit), phase= np.pi, amplitude=1 / 2
                    )
                else:
                    exp_rabi.play(
                        signal=f"drive_{qubit}", pulse=pi_pulse(qubit), phase=0, amplitude=0
                    )

            with exp_rabi.section(uid="readout_section", play_after="change_of_basis"):
                exp_rabi.play(signal="measure",
                              pulse=readout_pulse(qubit),
                              phase=qubit_parameters[qubit]['angle']
                              )
                exp_rabi.acquire(
                    signal="acquire",
                    handle=f"handle_{basis}",
                    kernel=kernel,
                )
            with exp_rabi.section(uid="delay"):
                exp_rabi.delay(signal="measure", time=200e-6)

    return exp_rabi


# %% Create Experiment and Signal Map

results = []

for basis in ['X', 'Y', 'Z']:
    ramsey = ramsey_exp(basis)

    ramsey.set_signal_map(signal_map_default)

    compiled_ramsey_exp = session.compile(ramsey)

    ramsey_results = session.run()

    acquire_results = ramsey_results.get_data(f"handle_{basis}")
    amplitudes = abs(acquire_results)
    results.append(amplitudes)

# %%
results = np.array(results)

r_results = np.sqrt((2 * results[0] - 1) ** 2 + (2 * results[1] - 1) ** 2 + (2 * results[2] - 1) ** 2)

x_axis = np.linspace(sweep_start, sweep_stop, step_num)

guess = [1, 1e-6, 0, 12e-6, np.mean(amplitudes)]

plt.plot(x_axis * 1e6, r_results, color='blue', marker='o', markersize=2)
# plt.plot(x_axis * 1e6, results[0], color='green', marker='o', markersize=2)
plt.plot(x_axis * 1e6, results[2], color='red', marker='o', markersize=2)


def cos_wave_exp(x, amplitude, T, phase, tau, offset):
    return amplitude * np.cos(2 * np.pi / T * x + phase) * np.exp(-x / tau) + offset


# fit
# params, params_covariance = curve_fit(
#     cos_wave_exp, x_axis, amplitudes, p0=guess)
#
# # plot
# plt.plot(x_axis * 1e6, cos_wave_exp(x_axis, *params), 'r-', label='Fitted Curve (T2 = %.3f us)' % (params[3] * 1e6))
# # #
# freq = 1 / params[1] * 1e-6
# T2 = params[3]

# text = f"T2: {T2*1e6:.4f} us"
# plt.text(0.65, 0.1, text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8))
if mode == 'disc':
    plt.ylim([0, 1])
plt.title(f"Ramesy Experiment title {qubit}")
plt.xlabel('Time [us]')
plt.ylabel('Amplitude [a.u.]')
plt.legend()
plt.show()

# %% save data
measured_data = {}


measured_data = dict(X=results[0], Y=results[1], Z=results[2], R=r_results)
sweep_parameters = dict(delay=np.array(x_axis))
units = dict(delay="s")

meta_data = dict( tags=["Nadav-Lab", "Ramsey_tomog"], user="Guy", qubit=qubit, qubit_parameters=qubit_parameters[qubit], comment = "protocol: Y/2 --- delay --- tomography")

exp_result = dict(measured_data=measured_data, sweep_parameters=sweep_parameters, units=units,
                  meta_data=meta_data)

lu.create_logfile("Ramsey_tomog", **exp_result, loop_type="1d")


# print(f"freq = {freq} MHz")
# print(f"T2 = {T2 * 1e6} us")

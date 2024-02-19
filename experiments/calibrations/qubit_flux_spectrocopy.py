from laboneq.simple import *
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from laboneq.contrib.example_helpers.plotting.plot_helpers import (
    plot_results,
    plot_simulation,
)

from exp_helper import *
from pulses import *
from qubit_parameters import qubit_parameters

from laboneq.contrib.example_helpers.data_analysis.data_analysis import fit_Spec
import scipy.optimize as opt

import json
from datetime import datetime
import os

from collections import OrderedDict

# %% parameters
qubit = "q1"
exp = initialize_exp()
device_setup = exp.create_device_setup(modulation_type='hardware')
exp_signals = exp.signals(qubit)
signal_map_default = exp.signal_map_default(qubit)

# %%

session = Session(device_setup=device_setup)
session.connect(do_emulation=False)

# %%

simulate = False
n_avg = 500
amp = 1 / 10  # ~0.4 (q3) for 2nd E level, 1/100 for 1st E level
w0 = True
plot_from_json = False

if w0:
    center = qubit_parameters[qubit]["qb_freq"]
else:
    center = qubit_parameters[qubit]["w125"]

span = 100e6
steps = 101

drive_LO = qubit_parameters[qubit]["qb_lo"]

dfs = np.linspace(start=center - span / 2, stop=center + span / 2, num=steps)  # for carrier calculation

freq_sweep = SweepParameter(uid=f'freq_sweep_{qubit}', values=dfs - drive_LO)  # sweep object


# %%


def qubit_spectroscopy(freq_sweep):
    exp_qspec = Experiment(
        uid="Qubit Spectroscopy",
        signals=exp_signals,
    )
    with exp_qspec.acquire_loop_rt(
            uid="freq_shots",
            count=n_avg,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
    ):
        with exp_qspec.sweep(uid="qfreq_sweep", parameter=freq_sweep):
            with exp_qspec.section(uid="qubit_excitation"):
                exp_qspec.play(signal=f"drive_{qubit}", pulse=spec_pulse(qubit), amplitude=amp,
                               marker={"marker1": {"enable": True}})

            with exp_qspec.section(
                    uid="readout_section", play_after="qubit_excitation",
            ):
                # play readout pulse on measure line
                exp_qspec.play(signal="measure",
                               pulse=readout_pulse(qubit),
                               phase=qubit_parameters[qubit]['angle'],

                               )

                # trigger signal data acquisition
                exp_qspec.acquire(
                    signal="acquire",
                    handle="qubit_spec",
                    length=qubit_parameters[qubit]["res_len"],
                    # kernel = kernels[qubit]
                )
            # delay between consecutive experiment
            with exp_qspec.section(uid="delay"):
                # relax time after readout - for qubit relaxation to groundstate and signal processing
                exp_qspec.delay(signal="measure", time=120e-6)

    return exp_qspec


# %% Run Exp
exp_calibration = Calibration()

exp_calibration[f"drive_{qubit}"] = SignalCalibration(
    oscillator=Oscillator(
        frequency=freq_sweep,
        modulation_type=ModulationType.HARDWARE,
    ),

)

# define experiment with frequency sweep for qubit 0
exp_qspec = qubit_spectroscopy(freq_sweep)
# apply calibration and signal map for qubit 0
exp_qspec.set_calibration(exp_calibration)
exp_qspec.set_signal_map(signal_map_default)
# %%
# compile the experiment on the open instrument session
compiled_qspec = session.compile(exp_qspec)

# %%plots
if simulate:
    plot_simulation(compiled_qspec, start_time=0, length=350e-6)

qspec_results = session.run(compiled_qspec)

# %% results

results = qspec_results.get_data("qubit_spec")

# %% set plotting variables
# **** don't forget to update file path

amplitude = np.abs(results)
# amplitude = correct_axis(amplitude,qubit_parameters[qubit]["ge"])

phase_radians = np.unwrap(np.angle(results))


def func_lorentz(x, width, pos, amp, off):
    return off + amp * width / (width ** 2 + (x - pos) ** 2)


guess = [1, 4.628e9 - 1000e6, -10, 0.0016]

popt = opt.curve_fit(func_lorentz, dfs, amplitude, p0=guess)[0]

resonanace_x = (qubit_parameters[qubit]["qb_freq"]) * 1e-6

anharmonicity = (qubit_parameters[qubit]["qb_freq"] - qubit_parameters[qubit]["w125"]) * 2 * 1e-9

drive_amp = qubit_parameters[qubit]['drive_amp'] * amp

flux_bias = qubit_parameters[qubit]['flux_bias']

new_min = dfs[np.argmin(amplitude)]

print(f"new min freq = {new_min * 1e-9} GHz")

if w0:
    plt.title(
        f'Qubit Spectroscopy {qubit} w0\n drive amp = {amp:5f} V \n flux bias = {flux_bias:.5f} V \nresonanance = {center * 1e-6} MHz  ',
        fontsize=18)
else:
    plt.title(
        f'Qubit Spectroscopy {qubit} w1\n drive amp = {amp:5f} V \n flux bias = {flux_bias:.5f} V \nresonanance = {center * 1e-6:.2f} MHz  ',
        fontsize=18)

plt.axvline(x=center * 1e-9, color='green', linestyle='--')
# center*1e-9
# plt.ylim([-0.2,1.2])
decimal_places = 4
plt.gca().xaxis.set_major_formatter(StrMethodFormatter(f"{{x:,.{decimal_places}f}}"))
# detuning plot
# plt.plot((dfs - center)*1e-6, amplitude, "k")
plt.plot(dfs * 1e-9, amplitude, "k")

# plt.plot(dfs*1e-6 - center*1e-6,func_lorentz(dfs,*popt))
plt.xlabel('Detuning [MHz]')
plt.ylabel('Amplitude [a.u.]')

print(f'anhrmonicity = {anharmonicity * 1e9} MHz')
print(f"width = {popt[0] * 1e-6} MHz")

# LabOne Q:
from laboneq.simple import *

# Helpers:
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
from qubit_parameters import *

# %% parameters
qubit = "q4"

mode = 'spec'
modulation_type = 'hardware' if mode == 'spec' else 'software'
if mode == 'spec':
    acquisition_type = AcquisitionType.SPECTROSCOPY
elif mode == 'int':
    acquisition_type = AcquisitionType.INTEGRATION
elif mode == 'disc':
    acquisition_type = AcquisitionType.DISCRIMINATION
kernel = readout_pulse(qubit) if mode == 'spec' else kernels[qubit]

exp = initialize_exp()
device_setup = exp.create_device_setup(modulation_type=modulation_type)
exp_signals = exp.signals(qubit)
signal_map_default = exp.signal_map_default(qubit)

# %%

session = Session(device_setup=device_setup)
session.connect(do_emulation=False)

# %% Quantum system parameters

simulate = False

sweep_start = 0
sweep_stop = 200e-6
step_num = 201

ts = np.linspace(0, sweep_stop, step_num)

delay_sweep = SweepParameter(values=ts)

exp_repetitions = 500


# %% Experiment Definition


def amplitude_rabi():
    exp_rabi = Experiment(
        uid="Amplitude Rabi",
        signals=exp_signals
    )

    ## define Rabi experiment pulse sequence
    # outer loop - real-time, cyclic averaging
    with exp_rabi.acquire_loop_rt(
            uid="rabi_shots",
            count=exp_repetitions,
            # averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=acquisition_type,
    ):
        # inner loop - real time sweep of Rabi ampitudes
        with exp_rabi.sweep(uid="rabi_sweep", parameter=delay_sweep):
            # play qubit excitation pulse - pulse amplitude is swept
            with exp_rabi.section(
                    uid="qubit_excitation"
            ):
                exp_rabi.play(
                    signal=f"drive_{qubit}", pulse=pi_pulse(qubit),
                )
                exp_rabi.delay(signal=f"measure", time=delay_sweep)
                # exp_rabi.delay(signal="drive", time=1e-6)
            # readout pulse and data acquisition
            with exp_rabi.section(uid="readout_section", play_after="qubit_excitation"):
                # play readout pulse on measure line
                exp_rabi.play(signal="measure",
                              pulse=readout_pulse(qubit),
                              phase=qubit_parameters[qubit]['angle'])
                # trigger signal data acquisition
                exp_rabi.acquire(
                    signal="acquire",
                    handle="amp_rabi",
                    kernel=kernel,
                    # kernel = kernels[qubit]
                )
            with exp_rabi.section(uid="delay"):
                # relax time after readout - for qubit relaxation to groundstate and signal processing
                exp_rabi.delay(signal="measure", time=150e-6)
    return exp_rabi


# %% Create Experiment and Signal Map

exp_rabi = amplitude_rabi()

exp_rabi.set_signal_map(signal_map_default)

# %%

compiled_rabi = session.compile(exp_rabi)

if simulate:
    plot_simulation(compiled_rabi, start_time=0, length=1e-6, signals=[f'drive_{qubit}', 'measure', 'acquire'])

# %% Run, Save, and Plot Results

rabi_results = session.run()

# %% plot parameters results KA


acquire_results = rabi_results.get_data("amp_rabi")
amplitudes = np.abs(acquire_results)


# amplitudes = correct_axis(amplitudes,qubit_parameters[qubit]["ge"])

# %% fitting

# Define the function to fit
def fit_function(t, Amplitude, T1, Const):
    return Amplitude * (1 - np.exp(-t / T1)) + Const


# Initial guess for the parameters
initial_guess = [0.1, 14.8e-6, 0.00018]

# Perform the curve fit
params, covariance = curve_fit(fit_function, ts, amplitudes, p0=initial_guess)

marker_size = 3
# Plot the original data and the fitted curve
plt.plot(ts * 1e6, amplitudes, 'bo', label='Original Data', markersize=marker_size)
plt.plot(ts * 1e6, fit_function(ts, *params), 'r-', label='Fitted Curve')
plt.xlabel('Time [usec]')
plt.ylabel('Amplitude')
plt.legend()
# text = f"Fitted Amplitude: {Amplitude_fit:.6f}\nFitted T1: {T1_fit:.6e}\nFitted Const: {Const_fit:.6f}"
# plt.text(0.5, 0.5, text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8))
plt.title(f'T1 Experiment \nFitted T1 = {params[1] * 1e6:.2f} us')

plt.show()

# Display the fitted parameters
print(f"Fitted T1 = {params[1] * 1e6} us")

# plt.show()

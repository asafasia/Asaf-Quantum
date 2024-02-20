

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

# %% parameters
qubit = "q2"

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

exp_repetitions = 700

w = 1e6
sweep_start = 0
sweep_stop = 20e-6
step_num = 500

sweep_time = LinearSweepParameter(uid="sweep", start=sweep_start, stop=sweep_stop, count=step_num)


# %% Experiment Definition


def ramsey_exp():
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
                    signal=f"drive_{qubit}", pulse=pi_pulse(qubit), amplitude=1 / 2)

                exp_rabi.delay(signal=f"drive_{qubit}", time=sweep_time)

                exp_rabi.play(
                    signal=f"drive_{qubit}", pulse=pi_pulse(qubit), phase=2 * np.pi * w * sweep_time, amplitude=1 / 2
                )

            with exp_rabi.section(uid="readout_section", play_after="qubit_excitation"):
                exp_rabi.play(signal="measure",
                              pulse=readout_pulse(qubit),
                              phase=qubit_parameters[qubit]['angle']
                              )
                exp_rabi.acquire(
                    signal="acquire",
                    handle="amp_rabi",
                    kernel=kernel,
                )
            with exp_rabi.section(uid="delay"):
                exp_rabi.delay(signal="measure", time=100e-6)

    return exp_rabi


# %% Create Experiment and Signal Map
ramsey_exp = ramsey_exp()

ramsey_exp.set_signal_map(signal_map_default)

# %% Compile
compiled_ramsey_exp = session.compile(ramsey_exp)
if simulate:
    plot_simulation(compiled_ramsey_exp, start_time=0, length=15e-6)

# %% Run, Save, and Plot Results
ramsey_results = session.run()

# %% plot
acquire_results = ramsey_results.get_data("amp_rabi")
amplitudes = abs(acquire_results)
# amplitudes = correct_axis(amplitudes, qubit_parameters[qubit]["ge"])

x_axis = np.linspace(sweep_start, sweep_stop, step_num)

guess = [1, 1e-6, 0, 12e-6, np.mean(amplitudes)]

plt.plot(x_axis * 1e6, amplitudes, color='blue', marker='o', markersize=2)

def cos_wave_exp(x, amplitude, T, phase, tau, offset):
    return amplitude * np.cos(2 * np.pi / T * x + phase) * np.exp(-x / tau) + offset


# fit
params, params_covariance = curve_fit(
    cos_wave_exp, x_axis, amplitudes, p0=guess)

# plot
plt.plot(x_axis * 1e6, cos_wave_exp(x_axis, *params), 'r-', label='Fitted Curve (T2 = %.3f us)' % (params[3] * 1e6))
#
freq = 1 / params[1] * 1e-6
T2 = params[3]

# text = f"T2: {T2*1e6:.4f} us"
# plt.text(0.65, 0.1, text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8))

plt.title(f"Ramesy Experiment {qubit}")
plt.xlabel('Time [us]')
plt.ylabel('Amplitude [a.u.]')
plt.legend()
plt.show()

print(f"freq = {freq} MHz")
print(f"T2 = {T2 * 1e6} us")

# %% save to labber
import labber.labber_util as lu
measured_data = dict(amplitude = np.array(acquire_results))
sweep_parameters = dict(delay = np.array(sweep_time.values))
units = dict(delay = 's')
meta_data = dict(user = "Guy", tags = [qubit,  'T2', 'Ramsey'], qubit_parameters = qubit_parameters)
exp_result = dict(measured_data=measured_data, sweep_parameters=sweep_parameters, units=units,
                  meta_data=meta_data)

lu.create_logfile("T2_ramsey", **exp_result, loop_type="1d")

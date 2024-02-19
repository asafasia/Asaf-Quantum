from laboneq.simple import *
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from qdac_test import play_flux_qdac

from laboneq.contrib.example_helpers.plotting.plot_helpers import (
    plot_results,
    plot_simulation,
)

from helper.exp_helper import *
from helper.kernels import kernels
from helper.pulses import *
from qubit_parameters import qubit_parameters

from laboneq.contrib.example_helpers.data_analysis.data_analysis import fit_Spec
import scipy.optimize as opt
from scipy.optimize import curve_fit

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
exp_signals = exp.signals(qubit)
signal_map_default = exp.signal_map_default(qubit)

# %%

session = Session(device_setup=device_setup)
session.connect(do_emulation=False)

# %%

simulate = False
n_avg = 100
amp = 1 / 20
w0 = True

if w0:
    center = qubit_parameters[qubit]["qb_freq"]
else:
    center = qubit_parameters[qubit]["w125"]

span = 200e6
steps = 101

drive_LO = qubit_parameters[qubit]["qb_lo"]

dfs = np.linspace(start=center - span / 2, stop=center + span / 2, num=steps)  # for carrier calculation

freq_sweep = SweepParameter(uid=f'freq_sweep_{qubit}', values=dfs - drive_LO)  # sweep object

min_flux = -1.5  # [V]
max_flux = 3  # [V]
qubit_flux_bias = qubit_parameters[qubit]['flux_bias']


def safe_flux(min_flux, max_flux):
    print(abs(qubit_flux_bias))
    if abs(min_flux) / abs(qubit_flux_bias) > 4:
        raise 'choose smaller min_flux'
    if abs(max_flux) / abs(qubit_flux_bias) > 4:
        raise 'choose smaller max_flux'
    else:
        print('all ok')


safe_flux(min_flux, max_flux)

flux_step_num = 30
flux_sweep = LinearSweepParameter(
    "flux_sweep", min_flux, max_flux, flux_step_num)


# %%


def session_flux(session, qubit, flux_bias):
    play_flux_qdac(qubit, flux_bias)


def qubit_spectroscopy(flux_sweep, freq_sweep):
    exp_qspec = Experiment(
        uid="Qubit Spectroscopy",
        signals=exp_signals,
    )

    with exp_qspec.sweep(uid="near-time sweep", parameter=flux_sweep, execution_type=ExecutionType.NEAR_TIME):
        exp_qspec.call(session_flux, qubit=qubit, flux_bias=flux_sweep)
        with exp_qspec.acquire_loop_rt(
                uid="freq_shots",
                count=n_avg,
                acquisition_type=acquisition_type,
        ):
            with exp_qspec.sweep(uid="qfreq_sweep", parameter=freq_sweep):
                with exp_qspec.section(uid="qubit_excitation"):
                    exp_qspec.play(signal=f"drive_{qubit}", pulse=spec_pulse(qubit), amplitude=amp)

                with exp_qspec.section(
                        uid="readout_section", play_after="qubit_excitation", trigger={"measure": {"state": True}}
                ):
                    # play readout pulse on measure line
                    exp_qspec.play(signal="measure",
                                   pulse=readout_pulse(qubit),
                                   phase=qubit_parameters[qubit]['angle'])
                    # trigger signal data acquisition
                    exp_qspec.acquire(
                        signal="acquire",
                        handle="qubit_spec",
                        kernel=kernel,
                        # kernel = kernel
                    )
                # delay between consecutive experiment
                with exp_qspec.section(uid="delay"):
                    # relax time after readout - for qubit relaxation to groundstate and signal processing
                    exp_qspec.delay(signal="measure", time=120e-6)

        return exp_qspec


# %% Run Exp
session.register_user_function(session_flux)
exp_calibration = Calibration()

exp_calibration[f"drive_{qubit}"] = SignalCalibration(
    oscillator=Oscillator(
        frequency=freq_sweep,
        modulation_type=ModulationType.HARDWARE,
    ),

)

# define experiment with frequency sweep for qubit 0
exp_qspec = qubit_spectroscopy(flux_sweep, freq_sweep)
# apply calibration and signal map for qubit 0
exp_qspec.set_calibration(exp_calibration)
exp_qspec.set_signal_map(signal_map_default)
# %%
# compile the experiment on the open instrument session
compiled_qspec = session.compile(exp_qspec)

# %%plots
if simulate:
    plot_simulation(compiled_qspec, start_time=0, length=3e-6, signals=[f'drive_{qubit}', f'measure', ])

# %%
qspec_results = session.run(compiled_qspec)

# %%

results = qspec_results.get_data("qubit_spec")
amplitude = np.abs(results)
# amplitude = correct_axis(amplitude,qubit_parameters[qubit]["ge"])
amplitude_T = amplitude.T

# print(amplitude_T)

# plt.plot(dfs,amplitude_T)


# %% save results
import json
from datetime import datetime
import os

from collections import OrderedDict

experiment_name = "qubit_vs_flux_spec"
timestamp = time.strftime("%Y%m%dT%H%M%S")
Path("Results").mkdir(parents=True, exist_ok=True)
session.save_results(f"Results/{timestamp}_qb_flux_results_{qubit}.json")
print(f"File saved as Results/{timestamp}__qb_flux_results_{qubit}.json.json")

current_date = datetime.now().strftime("%Y-%m-%d")
# Create a folder based on the current date if it doesn't exist
folder_path = os.path.join("C:/Users/stud/Documents/GitHub/qhipu-files/LabOne Q/Exp_results", f"{current_date}_results")
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Generate a unique filename based on the current time
current_time = datetime.now().strftime("Time_%H-%M-%S")
filename = os.path.join(folder_path, f"{experiment_name}_{current_time}.json")

session.save_results(filename)

print(f"Data saved to {filename}")

# %% plot


# plt.plot(dfs, amplitude[2, :])
# plt.show()
# Plot the pcolormesh after the other plots
# %%
# if qubit=="q5":
#     flux_sweep=2*flux_sweep

x, y = np.meshgrid(flux_sweep.values, dfs)

plt.pcolormesh(x * 1000, y * 1e-6, amplitude_T, shading='auto', cmap='viridis', vmin=amplitude_T.min(),
               vmax=amplitude_T.max())

print(x)
plt.colorbar()
# Set labels and title
plt.xlabel('Flux [mV]')
plt.ylabel('Drive Frequency [MHz]')
# plt.xlim([-200, 200])
# plt.axhline(y=4.725e9, color='red', linestyle='--')
# plt.axvline(x=x_values[0], color='red', linestyle='--')
plt.legend()
plt.title(f'Qubit spectroscopy vs. Flux {qubit}', fontsize=18)

# Show the plot


max_fs = []
for amp in amplitude:
    max_freq = dfs[np.argmin(amp)]
    max_fs.append(max_freq)

max_fs = np.array(max_fs)

# plt.plot(flux_sweep.values*1e3,max_fs*1e-6,color = 'blue',label = 'max_points')
qubit_freq = qubit_parameters[qubit]['qb_freq']
plt.axhline(y=qubit_freq * 1e-6, color='blue', linestyle='--', label='qb freq')


def parbola(x, a, b, c):
    return a * x ** 2 + b * x + c


# params = curve_fit(parbola, flux_sweep.values, max_fs, p0=[100,100,4e9])[0]

# plt.plot(flux_sweep.values*1e3, parbola(flux_sweep.values, *params)*1e-6,label = 'fit',color = 'red')

plt.legend()

plt.show()

# y0 = 4.4e9

# plt.axvline(y = y0*1e-6)
# %% save to labber
import labber.labber_util as lu
measured_data = dict(amplitude = np.array(results))
sweep_parameters = dict(frequency = np.array(dfs), flux = np.array(flux_sweep.values))
units = dict(frequency = 'Hz', flux = 'V')
meta_data = dict(user = "Guy", tags = [qubit,  'spectroscopy', 'flux_spectroscopy'], qubit_parameters = qubit_parameters)
exp_result = dict(measured_data=measured_data, sweep_parameters=sweep_parameters, units=units,
                  meta_data=meta_data)

lu.create_logfile("spectroscopy_w_flux_scan", **exp_result, loop_type="2d")


# %% update abc


# user_input = input("Do you want to update the pi amplitude? [y/n]")
#
# if user_input == 'y':
#     update_qp(qubit, 'abc', list(params))
#     print('updated !!!')
#
# elif user_input == 'n':
#     print('not_updated')
# else:
#     raise Exception("Invalid input")

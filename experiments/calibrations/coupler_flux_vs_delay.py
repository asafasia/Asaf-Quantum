from helper.utility_functions import correct_axis

import matplotlib.pyplot as plt
from laboneq.contrib.example_helpers.plotting.plot_helpers import (
    plot_simulation,
)
from laboneq.simple import *
import numpy as np

from helper import pulses, exp_helper, utility_functions
from helper.plotter import Plotter
from qubit_parameters import qubit_parameters
from helper.experiment_results import ExperimentData

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
    qubit_m = "q3"
    qubit_s = "q4"
    n_port = 7
else:
    raise ValueError('Coupler not found')

exp = exp_helper.initialize_exp()
device_setup = exp.create_device_setup(modulation_type='hardware')
exp_signals = exp.signals(qubit_s)
signal_map_default = exp.signal_map_default(qubit_s)

# %%
session = Session(device_setup=device_setup)
session.connect(do_emulation=False)

# %% parameters for user
central_frequency = qubit_parameters[qubit_s]["qb_freq"]
simulate = False

# %%parameters for sweeping
exp_repetitions = 5000

# parameters for flux sweep:
min_flux = 150e-3  # [V]
max_flux = -400e-3  # [V]
flux_step_num = 220

# parameters for pulse length sweep:
min_time = 0  # [sec]
max_time = 2e-6  # [sec]
time_step_num = 220

# %% parameters for experiment

flux_sweep = LinearSweepParameter(
    "flux_sweep", min_flux, max_flux, flux_step_num)
time_sweep = LinearSweepParameter(
    "freq_RF_sweep", min_time, max_time, time_step_num)


# %%
def qubit_spectroscopy(flux_sweep, time_sweep):
    # Create qubit spectroscopy Experiment - uses qubit drive, readout drive and data acquisition lines
    exp_qspec = Experiment(
        uid="Qubit Spectroscopy",
        signals=[
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
            ExperimentSignal(f"drive_{qubit_m}"),
            ExperimentSignal(f"drive_{qubit_s}"),
        ]

    )
    hdawg_address = device_setup.instrument_by_uid("device_hdawg").address

    with exp_qspec.sweep(uid="near-time sweep", parameter=flux_sweep, execution_type=ExecutionType.NEAR_TIME):
        exp_qspec.set_node(path=f"/{hdawg_address}/sigouts/{n_port}/offset", value=flux_sweep)
        with exp_qspec.acquire_loop_rt(
                uid="freq_shots",
                count=exp_repetitions,
                acquisition_type=AcquisitionType.SPECTROSCOPY,
                averaging_mode=AveragingMode.CYCLIC

        ):
            with exp_qspec.sweep(uid="qfreq_sweep", parameter=time_sweep):
                with exp_qspec.section(uid="qubit_excitation"):
                    exp_qspec.play(signal=f"drive_{qubit_m}", pulse=pulses.pi_pulse(qubit_m))
                    exp_qspec.play(signal=f"drive_{qubit_s}", pulse=pulses.pi_pulse(qubit_s))

                with exp_qspec.section(uid="time_delay", play_after="qubit_excitation"):
                    exp_qspec.delay(signal="measure", time=time_sweep)

                with exp_qspec.section(uid="readout_section", play_after="time_delay"):
                    exp_qspec.play(signal="measure", pulse=pulses.readout_pulse(qubit_s))

                    exp_qspec.acquire(
                        signal="acquire",
                        handle="qubit_spec",
                        length=qubit_parameters[qubit_s]["res_len"],
                    )
                with exp_qspec.section(uid="delay"):
                    # relax time after readout - for qubit relaxation to groundstate and signal processing
                    exp_qspec.delay(signal="measure", time=120e-6)
    return exp_qspec


# %% Run Exp
# define experiment with frequency sweep for qubit 0
exp_qspec = qubit_spectroscopy(flux_sweep, time_sweep)

signal_map_default = {
    f"drive_{qubit_s}": device_setup.logical_signal_groups[qubit_s].logical_signals["drive_line"],
    f"drive_{qubit_m}": device_setup.logical_signal_groups[qubit_m].logical_signals["drive_line"],
    "measure": device_setup.logical_signal_groups[qubit_s].logical_signals["measure_line"],
    "acquire": device_setup.logical_signal_groups[qubit_s].logical_signals["acquire_line"]
}

# apply calibration and signal map for qubit 0
exp_qspec.set_signal_map(signal_map_default)
# %% compile exp
# compile the experiment on the open instrument session

if simulate:
    compiled_qspec = session.compile(exp_qspec)
    plot_simulation(compiled_qspec, start_time=0, length=300e-6)

# %% run the compiled experiment
qspec_results = session.run(exp_qspec)

# %%

acquire_results = qspec_results.get_data("qubit_spec")
amplitude = np.real(acquire_results)
amplitude = correct_axis(amplitude, qubit_parameters[qubit_s]["ge"])

index = 0
print(time_sweep.values.shape)
print('Current flux: ', flux_sweep.values[index])

plt.title(f'CPhase oscillation vs. time \nflux point = {flux_sweep.values[index]} V')
plt.plot(time_sweep.values * 1e6, amplitude[index, :])
plt.ylabel('Amplitude [a.u.]')
plt.xlabel('time [us]')
plt.show()

# %% plot 2d

amplitude = amplitude.T

x, y = np.meshgrid(flux_sweep.values, time_sweep.values)

plt.xlabel('Flux [mV]')
plt.ylabel('time [us]')
plt.title(f'Qubit Spectroscopy vs. Coupler Flux {qubit_s} {coupler}', fontsize=18)

plt.pcolormesh(x * 1e3, y * 1e6, amplitude)

plt.colorbar()
plt.show()

# %% FFT
signal = amplitude - np.mean(amplitude)

# calculate FFT for each flux value
bias_duration = time_sweep.values
dt = bias_duration[1] - bias_duration[0]

bias_level = flux_sweep.values

bias_freq = np.fft.fftfreq(len(bias_duration), dt)
bias_freq = np.fft.fftshift(bias_freq)

Fsignal = np.zeros_like(signal)
for i in range(len(signal[0, :])):
    Fsignal[:, i] = abs(np.fft.fft(signal[:, i]))
    Fsignal[:, i] = np.fft.fftshift(Fsignal[:, i])

normalized = Fsignal / Fsignal.max()
gamma = 1 / 4
corrected = np.power(normalized, gamma)  # try values between 0.5 and 2 as a start point

# plt.axvline(x=565, color='green', linestyle='--')
fig_fft = plt.pcolormesh(bias_level * 1e3, abs(bias_freq) * 1e-6, corrected, cmap='coolwarm')
plt.title(f'FFT Qubit Spectroscopy vs. Coupler Flux  Coupler {coupler}', fontsize=18)

plt.xlabel('coupler bias [mV]')
plt.ylabel('bias frequency [MHz]')
plt.colorbar()
plt.show()

# %% save json file
# todo: solve problems with many qubits experiments
meta_data = {
    'type': '2d',
    'plot_properties':
        {
            'x_label': 'flux',
            'y_label': 'time',
            'z_label': 'Amplitude [a.u.]',
            'title': 'Controlled Phase oscillation vs. time'
        },
    'experiment_properties':
        {
            'slave_qubit': qubit_s,
            'master_qubit': qubit_m,
            'exp_repetitions': exp_repetitions

        },
    'qubit_parameters': qubit_parameters
}

data = {
    'x_data': time_sweep.values,
    'y_data': flux_sweep.values,
    'z_data': amplitude
}

exp_data = ExperimentData(meta_data=meta_data, data=data)
exp_data.save_data(file_name='coupler_flux_vs_delay_2d')

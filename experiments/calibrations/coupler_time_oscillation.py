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
exp_repetitions = 1000

flux = -0.14  # [V]
max_time = 0.4e-6  # [sec]
time_step_num = 150

# %% parameters for experiment
time_sweep = LinearSweepParameter(
    "freq_RF_sweep", 16e-9, max_time, time_step_num)

# %%
def qubit_spectroscopy(time_sweep):
    # Create qubit spectroscopy Experiment - uses qubit drive, readout drive and data acquisition lines
    exp_qspec = Experiment(
        uid="Qubit Spectroscopy",
        signals=[
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
            ExperimentSignal(f"drive_{qubit_m}"),
            ExperimentSignal(f"drive_{qubit_s}"),
            ExperimentSignal(f"flux_c43"),
        ]

    )

    with exp_qspec.acquire_loop_rt(
            uid="freq_shots",
            count=exp_repetitions,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
            averaging_mode=AveragingMode.CYCLIC

    ):
        with exp_qspec.sweep(uid="qfreq_sweep", parameter=time_sweep):
            with exp_qspec.section(uid="qubit_excitation"):
                # exp_qspec.play(signal=f"drive_{qubit_m}", pulse=pulses.pi_pulse(qubit_m))
                exp_qspec.play(signal=f"drive_{qubit_s}", pulse=pulses.pi_pulse(qubit_s))

            with exp_qspec.section(uid="flux"):
                exp_qspec.play(signal=f"flux_c43", pulse=pulses.flux_pulse(qubit_s, amplitude=flux), length=time_sweep)

            with exp_qspec.section(uid="readout_section", play_after="flux"):
                exp_qspec.play(signal="measure", pulse=pulses.readout_pulse(qubit_s))

                exp_qspec.acquire(
                    signal="acquire",
                    handle="qubit_spec",
                    length=qubit_parameters[qubit_s]["res_len"],
                )
            with exp_qspec.section(uid="delay"):
                exp_qspec.delay(signal="measure", time=120e-6)

    return exp_qspec


# %% Run Exp
# define experiment with frequency sweep for qubit 0
exp_qspec = qubit_spectroscopy(time_sweep)

signal_map_default = {
    f"drive_{qubit_s}": device_setup.logical_signal_groups[qubit_s].logical_signals["drive_line"],
    f"drive_{qubit_m}": device_setup.logical_signal_groups[qubit_m].logical_signals["drive_line"],
    "flux_c43": device_setup.logical_signal_groups["c43"].logical_signals["flux_line"],
    "measure": device_setup.logical_signal_groups[qubit_s].logical_signals["measure_line"],
    "acquire": device_setup.logical_signal_groups[qubit_s].logical_signals["acquire_line"]
}

# apply calibration and signal map for qubit 0
exp_qspec.set_signal_map(signal_map_default)
# %% compile exp
# compile the experiment on the open instrument session

if simulate:
    compiled_qspec = session.compile(exp_qspec)
    plot_simulation(compiled_qspec, start_time=0, length=5e-6)

# %% run the compiled experiment
qspec_results = session.run(exp_qspec)

# %%
acquire_results = qspec_results.get_data("qubit_spec")
amplitude = np.real(acquire_results)
amplitude = correct_axis(amplitude, qubit_parameters[qubit_s]["ge"])

print('Current flux: ', flux)

plt.title(f'CPhase oscillation vs. time \nflux point = {flux} V')
plt.plot(time_sweep.values * 1e6, amplitude)
plt.ylabel('Amplitude [a.u.]')
plt.xlabel('time [us]')
plt.show()

# %% save json file
meta_data = {
    'type': '1d',
    'plot_properties':
        {
            'x_label': 'time',
            'y_label': 'Amplitude [a.u.]',
            'title': 'Controlled Phase oscillation vs. time'
        },
    'experiment_properties':
        {
            'slave_qubit': qubit_s,
            'master_qubit': qubit_m,
            'flux station': flux,
            'state': '01',
            'exp_repetitions': exp_repetitions

        },
    'qubit_parameters': qubit_parameters
}

data = {
    'x_data': time_sweep.values * 1e6,
    'y_data': amplitude,
}

exp_data = ExperimentData(meta_data=meta_data, data=data)
exp_data.save_data(file_name='coupler_flux_vs_delay_1d')

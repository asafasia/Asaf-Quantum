from laboneq.contrib.example_helpers.plotting.plot_helpers import (
    plot_simulation,
)
from laboneq.simple import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from helper import pulses, exp_helper
from helper.utility_functions import cos_wave, correct_axis
from helper.experiment_results import ExperimentData
from helper.plotter import Plotter
from qubit_parameters import qubit_parameters, update_qp

# %% parameters
qubit = "q4"

exp = exp_helper.initialize_exp()
device_setup = exp.create_device_setup(modulation_type='hardware')

exp_signals = exp.signals(qubit)
signal_map_default = exp.signal_map_default(qubit)

# %%

session = Session(device_setup=device_setup)
session.connect(do_emulation=False)

# %% amplitude sweep
simulate = False

time_min = 32e-9  # 0.5*8e-3 # [V]
time_max = 1e-6  # 0.8 # [V]
time_num = 50

exp_repetitions = 1000

sweep_rabi_time = LinearSweepParameter(start=time_min, stop=time_max, count=time_num)


# %% Experiment Definition

# function that returns an amplitude Rabi experiment


def amplitude_rabi(time_sweep):
    exp_rabi = Experiment(
        uid="Amplitude Rabi",
        signals=[
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
            ExperimentSignal(f"drive_{qubit}"),
        ]
    )

    # define Rabi experiment pulse sequence
    # outer loop - real-time, cyclic averaging
    with exp_rabi.acquire_loop_rt(
            uid="rabi_shots",
            count=exp_repetitions,
            # averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
    ):
        # inner loop - real time sweep of Rabi ampitudes
        with exp_rabi.sweep(uid="rabi_sweep", parameter=time_sweep):
            # play qubit excitation pulse - pulse amplitude is swept
            with exp_rabi.section(
                    uid="qubit_excitation"
            ):
                exp_rabi.play(signal=f"drive_{qubit}", pulse=pulses.pi_pulse(qubit), length=time_sweep)
            # readout pulse and data acquisition
            with exp_rabi.section(uid="readout_section", play_after="qubit_excitation"):
                # play readout pulse on measure line
                exp_rabi.play(signal="measure", pulse=pulses.readout_pulse(qubit))
                # trigger signal data acquisition
                exp_rabi.acquire(
                    signal="acquire",
                    handle="amp_rabi",
                    length=qubit_parameters[qubit]["res_len"],
                )
            with exp_rabi.section(uid="delay"):
                # relax time after readout - for qubit relaxation to groundstate and signal processing
                exp_rabi.delay(signal="measure", time=120e-6)
    return exp_rabi


# %% 

exp_rabi = amplitude_rabi(sweep_rabi_time)

signal_map_default = {"drive_q4": device_setup.logical_signal_groups[qubit].logical_signals["drive_line"],
                      "measure": device_setup.logical_signal_groups[qubit].logical_signals["measure_line"],
                      "acquire": device_setup.logical_signal_groups[qubit].logical_signals["acquire_line"]}

exp_rabi.set_signal_map(signal_map_default)

# %%
compiler_settings = {"SHFSG_MIN_PLAYWAVE_HINT": 1024, "SHFSG_MIN_PLAYZERO_HINT": 512}

compiled_rabi = session.compile(exp_rabi,compiler_settings=compiler_settings)

if simulate:
    plot_simulation(compiled_rabi, start_time=0, length=15e-6)

# %% Run, Save, and Plot results

rabi_results = session.run()

# %% plot results KA
acquire_results = rabi_results.get_data("amp_rabi")

amplitude = np.abs(acquire_results)
# phase_radians = np.unwrap(np.angle(acquire_results))


# Plot the amplitude in the first subplot
plt.plot(sweep_rabi_time.values, amplitude, '-', color='black', label='data')

plt.show()

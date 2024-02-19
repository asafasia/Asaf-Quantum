from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
from laboneq.simple import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from helper import pulses, exp_helper
from helper.utility_functions import correct_axis, cos_wave_exp
from helper.experiment_results import ExperimentData
from helper.plotter import Plotter
from qubit_parameters import qubit_parameters

# %% parameters
qubit = "q1"
exp = exp_helper.initialize_exp()
device_setup = exp.create_device_setup()
signal_map_default = exp.signal_map_default(qubit)
exp_signals = exp.signals(qubit)

# %%
session = Session(device_setup=device_setup)
session.connect(do_emulation=False)
signal_map_default = {
    f"drive_{qubit}": device_setup.logical_signal_groups[qubit].logical_signals["drive_line"],
    "measure": device_setup.logical_signal_groups[qubit].logical_signals["measure_line"],
    "acquire": device_setup.logical_signal_groups[qubit].logical_signals["acquire_line"]
}

# %% parameters
simulate = False

exp_repetitions = 2000

w = 1e6
sweep_start = 0
sweep_stop = 6e-6
step_num = 150

sweep_time = LinearSweepParameter(uid="sweep", start=sweep_start, stop=sweep_stop, count=step_num)


# %% Experiment Definition
def ramsey_exp():
    ramsey_exp = Experiment(
        uid="Amplitude Rabi",
        # signals=exp_signals
        signals=[
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
            ExperimentSignal(f"drive_{qubit}"),
        ]
    )
    # 

    with ramsey_exp.acquire_loop_rt(
            uid="rabi_shots",
            count=exp_repetitions,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
    ):
        with ramsey_exp.sweep(uid="rabi_sweep", parameter=sweep_time):
            with ramsey_exp.section(
                    uid="qubit_excitation"
            ):
                ramsey_exp.play(
                    signal=f"drive_{qubit}", pulse=pulses.pi_pulse(qubit), amplitude=1 / 2)

                ramsey_exp.delay(signal=f"drive_{qubit}", time=sweep_time)

                ramsey_exp.play(
                    signal=f"drive_{qubit}", pulse=pulses.pi_pulse(qubit), phase=2 * np.pi * w * sweep_time,
                    amplitude=1 / 2
                )

            with ramsey_exp.section(uid="readout_section", play_after="qubit_excitation"):
                ramsey_exp.play(signal="measure",
                                pulse=pulses.readout_pulse(qubit),
                                phase=qubit_parameters[qubit]['angle']
                                )
                ramsey_exp.acquire(
                    signal="acquire",
                    handle="amp_rabi",
                    kernel=pulses.readout_pulse(qubit),
                )
            with ramsey_exp.section(uid="delay"):
                ramsey_exp.delay(signal="measure", time=100e-6)

    return ramsey_exp


# %% Create Experiment and Signal Map
ramsey_exp = ramsey_exp()

ramsey_exp.set_signal_map(signal_map_default)

# %% Compile
compiled_ramsey_exp = session.compile(ramsey_exp)
if simulate:
    plot_simulation(compiled_ramsey_exp, start_time=0, length=15e-6)

# %% Run, Save, and Plot results
ramsey_results = session.run()

# %% plot
acquire_results = ramsey_results.get_data("amp_rabi")

amplitude = acquire_results.real
amplitude = correct_axis(amplitude, qubit_parameters[qubit]["ge"])

x_axis = np.linspace(sweep_start, sweep_stop, step_num)

guess = [1, 1e-6, 0, 1e-6, np.mean(amplitude)]

params, params_covariance = curve_fit(cos_wave_exp, x_axis, amplitude, p0=guess)

plt.plot(x_axis * 1e6, amplitude, color='blue', marker='o', markersize=2)
plt.plot(x_axis * 1e6, cos_wave_exp(x_axis, *params), 'r-', label='Fitted Curve')

plt.title(f"Ramesy Experiment {qubit}")
plt.xlabel('Time [us]')
plt.ylabel('Amplitude [a.u.]')
plt.legend()
plt.show()

freq = 1 / params[1] * 1e-6
T2 = params[3]
qubit_freq = qubit_parameters[qubit]['qb_freq'] * 1e-6
flux_bias = qubit_parameters[qubit]['flux_bias']

print(f"freq = {freq} MHz")
print(f"T2 = {T2 * 1e6} us")

# %%
meta_data = {
    'type': '1d',
    'plot_properties':
        {
            'x_label': 'Qubit Frequency [MHz]',
            'y_label': 'Amplitude [a.u.]',
            'title': 'Qubit Spectroscopy Experiment'
        },
    'experiment_properties':
        {
            'qubit': qubit,
            'repetitions': exp_repetitions,
        },
    'qubit_parameters': qubit_parameters,

}

data = {
    'x_data': x_axis,
    'y_data': amplitude,
}

exp_data = ExperimentData(meta_data=meta_data, data=data)
exp_data.save_data(file_name=f'ramsey_{qubit}')

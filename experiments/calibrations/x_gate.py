import matplotlib.pyplot as plt
from laboneq.contrib.example_helpers.plotting.plot_helpers import (
    plot_simulation,
)
from laboneq.simple import *
import numpy as np
from scipy.optimize import curve_fit

from helper import pulses, exp_helper, utility_functions
from helper.plotter import Plotter
from qubit_parameters import qubit_parameters
from helper.experiment_results import ExperimentData
from scipy.optimize import curve_fit

# %% devise setup
qubit = "q1"

exp = exp_helper.initialize_exp()
device_setup = exp.create_device_setup()
signal_map_default = exp.signal_map_default(qubit)
exp_signals = exp.signals(qubit)
# %%
session = Session(device_setup=device_setup)
session.connect(do_emulation=False)

# %% Experiment Parameters
simulate = False
points = 50
exp_repetitions = 10000


# %% Experiment Definition
def x_gate_rep():
    exp_rabi = Experiment(
        uid="Amplitude Rabi",
        signals=[
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
            ExperimentSignal(f"drive_{qubit}"),
        ]
    )

    with exp_rabi.acquire_loop_rt(
            uid="rabi_shots",
            count=exp_repetitions,
            averaging_mode=AveragingMode.CYCLIC,

            acquisition_type=AcquisitionType.SPECTROSCOPY):

        for i in range(points):
            with exp_rabi.section(alignment=SectionAlignment.RIGHT):
                for i in range(i):
                    exp_rabi.play(signal=f"drive_{qubit}",
                                  pulse=pulses.pi_pulse(qubit), amplitude=1 / 2, )

            with exp_rabi.section():
                exp_rabi.reserve(f"drive_{qubit}")

                exp_rabi.play(signal="measure",
                              pulse=pulses.readout_pulse(qubit),
                              phase=qubit_parameters[qubit]['angle'])

                exp_rabi.acquire(
                    signal="acquire",
                    handle="amp_rabi",
                    length=qubit_parameters[qubit]["res_len"],

                )
            with exp_rabi.section(uid="delay"):
                exp_rabi.reserve(f"drive_{qubit}")

                exp_rabi.delay(signal="measure", time=100e-6)

    return exp_rabi


# %% create experiment
x_gate_rep = x_gate_rep()
signal_map_default = {f"drive_{qubit}": device_setup.logical_signal_groups[qubit].logical_signals["drive_line"],
                      "measure": device_setup.logical_signal_groups[qubit].logical_signals["measure_line"],
                      "acquire": device_setup.logical_signal_groups[qubit].logical_signals["acquire_line"]}
x_gate_rep.set_signal_map(signal_map_default)

# %% Compile
compiled_rabi = session.compile(x_gate_rep)
if simulate:
    plot_simulation(compiled_rabi, start_time=0, length=1e-3)

# %% plot
rabi_results = session.run()

acquire_results = rabi_results.get_data("amp_rabi")
amplitude = acquire_results.real
amplitude = utility_functions.correct_axis(amplitude, qubit_parameters[qubit]["ge"])
amp_mean = np.mean(amplitude)
pi_amp = qubit_parameters[qubit]['pi_amp']

plt.title(f'X Gate Repetitions Experimnet {qubit}')
plt.axhline(y=0.5, color='black')
plt.plot(range(points), amplitude, '.')
plt.plot(range(points)[1:-1:2], amplitude[1:-1:2])



plt.plot(range(points), amplitude)
plt.xlabel('X90 Gate Number')
plt.ylabel('Amplitude [a.u.]')
plt.text(points - 1 / 2 * points, max(amplitude) * 1, f'drive amp = {pi_amp:.4f} Volt')
plt.show()


# vec = amplitude[1:-1:2]




# args = curve_fit('cos', range(len(vec)), vec, p0=[1, 1, 1, 1])


# %% save
meta_data = {
    'type': '1d',
    'plot_properties':
        {
            'x_label': 'X90 Gate Number',
            'y_label': 'Amplitude [a.u.]',
            'title': 'T1 Experiment'
        },
    'experiment_properties':
        {
            'qubit': qubit,
            'repetitions': exp_repetitions,
        },
    'qubit_parameters': qubit_parameters[qubit],
}

data = {
    'x_data': list(range(points)),
    'y_data': amplitude
}

exp_data = ExperimentData(meta_data=meta_data, data=data)
exp_data.save_data(file_name='x_gate')

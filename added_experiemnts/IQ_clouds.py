from laboneq.simple import *
from numpy.typing import NDArray
from zhinst.utils.shfqa.multistate import QuditSettings
from laboneq.dsl.experiment import pulse_library as pl

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import FuncFormatter

from helper.kernels import kernels
from helper.exp_helper import *
from helper.pulses import *
from qubit_parameters import *

from laboneq.contrib.example_helpers.plotting.plot_helpers import (
    plot_results,
    plot_simulation,
)

from laboneq.contrib.example_helpers.generate_example_datastore import (
    generate_example_datastore,
    get_first_named_entry,
)

from pprint import pprint

from laboneq.contrib.example_helpers.feedback_helper import (
    state_emulation_pulse,
    create_calibration_experiment,
    create_discrimination_experiment,
)

from pulses import *
from qubit_parameters import qubit_parameters, update_qp

# %% devise setup
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

# %%


exp = initialize_exp()
device_setup = exp.create_device_setup(modulation_type)
exp_signals = exp.signals(qubit)
signal_map_default = exp.signal_map_default(qubit)

# %% parameters
do_emulation = False
simulate = True
show_clouds = True
exp_repetitions = 5000
plot_from_json = False

# %%
session = Session(device_setup=device_setup)
session.connect(do_emulation=do_emulation, reset_devices=True)

# %% expetiment
exp_0 = Experiment(
    uid="Optimal weights",
    signals=exp_signals
)

with exp_0.acquire_loop_rt(
        uid="shots",
        count=exp_repetitions,  # pow(2, average_exponent),
        averaging_mode=AveragingMode.SINGLE_SHOT,
        acquisition_type=acquisition_type,
):
    with exp_0.section():
        exp_0.play(signal=f"drive_{qubit}", pulse=pi_pulse(qubit), amplitude=0)

    with exp_0.section():
        exp_0.play(signal="measure",
                   pulse=readout_pulse(qubit),
                   phase=qubit_parameters[qubit]['angle'])
        exp_0.acquire(
            signal="acquire", handle="ac_0", kernel=kernel,
        )

    with exp_0.section():
        exp_0.delay(signal="measure", time=200e-6)

exp_1 = Experiment(
    uid="Optimal weights",
    signals=exp_signals
)

with exp_1.acquire_loop_rt(
        uid="shots",
        count=exp_repetitions,  # pow(2, average_exponent),
        averaging_mode=AveragingMode.SINGLE_SHOT,
        acquisition_type=acquisition_type,
):
    with exp_1.section(uid='drive'):
        exp_1.play(signal=f"drive_{qubit}", pulse=pi_pulse(qubit), amplitude=1)

    with exp_1.section(uid='measure'):
        exp_1.reserve(signal=f"drive_{qubit}")

        exp_1.play(signal="measure",
                   pulse=readout_pulse(qubit),
                   phase=qubit_parameters[qubit]['angle']
                   )

        exp_1.acquire(
            signal="acquire",
            handle="ac_1",
            kernel=kernel,
        )

    with exp_1.section():
        exp_1.delay(signal="measure", time=200e-6)

exp_0.set_signal_map(signal_map_default)
exp_1.set_signal_map(signal_map_default)

# %% run the first experiment and access the data
results_0 = session.run(exp_0)
results_1 = session.run(exp_1)

# %%
import cmath

raw_0 = results_0.get_data("ac_0")
raw_1 = results_1.get_data("ac_1")

traces = np.array([raw_0, raw_1])

m1 = np.mean(traces[0])
m2 = np.mean(traces[1])
v1 = np.var(traces[0])
v2 = np.var(traces[1])

angle = -cmath.phase(m2 - m1)
seperation = abs((m2 - m1) / np.sqrt(v1))

threshold = (m1.real + m2.real) / 2

print("threshold", threshold)
print(f"ground = {m1.real} \nexceited = {m2.real}")
print(f"angle = {angle + qubit_parameters[qubit]['angle']}")

if not do_emulation and mode == 'int':
    update_qp(qubit, 'threshold', threshold)
    update_qp(qubit, 'angle', angle + qubit_parameters[qubit]['angle'])
    # update_qp(qubit, 'ge', [m1.real,m2.real])


# %% x axis number formatting

# Define a custom formatter function
def custom_formatter(value, _):
    if value < 1:
        return f"{value * 1e3:.2f}"  # Convert to milli and show one decimal place
    else:
        return f"{value * 1e-3:.0e}"  # Convert to kilo and use scientific notation


# Apply the custom formatter to the x-axis


# %% clouds
if show_clouds:
    plt.plot(traces[0].real, traces[0].imag, '.', alpha=0.6, label='ground')
    plt.plot(traces[1].real, traces[1].imag, '.', alpha=0.6, label='excited')

    # mean
    plt.plot(m1.real, m1.imag, 'x', color='black')
    plt.plot(m2.real, m2.imag, 'x', color='black')

    plt.title(f'IQ Clouds {qubit}')
    plt.xlabel('I [a.u.]', )
    plt.ylabel('Q [a.u.]')
    plt.legend()
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(custom_formatter))
    plt.show()

# %% histogram

res_freq = qubit_parameters[qubit]['res_freq']
res_amp = qubit_parameters[qubit]['res_amp']
res_len = qubit_parameters[qubit]['res_len']

plt.title(
    f"Seperation Histogram {qubit} \nSeperation = {seperation:.3f} \nres_freq = {res_freq * 1e-6:.2f} MHz  \nres amp = {res_amp} V \nres len = {res_len * 1e6} us ")
plt.hist(traces[0].real, bins=80, edgecolor='black', label='ground state', alpha=0.6)
plt.hist(traces[1].real, bins=80, edgecolor='black', label='excited state', alpha=0.5)
# plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_formatter))


plt.xlabel('I [a.u.]')
plt.ylabel('number')
plt.legend()
plt.show()
# plt.axvline(x=m2/2 + m1/2,linestyle='--',color='black')
# plt.axvline(x=m1,linestyle='--',color='blue')
# plt.axvline(x=m2,linestyle='--',color='red')

import matplotlib.pyplot as plt
from laboneq.contrib.example_helpers.randomized_benchmarking_helper import (
    clifford_parametrized,
    generate_play_rb_pulses
)
from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
from scipy.optimize import curve_fit

from helper.exp_helper import *
from helper import pulses
import numpy as np

from helper.experiment_results import ExperimentData
from qubit_parameters import qubit_parameters
from helper.utility_functions import correct_axis


# %% devise setup
qubit = "q4"
exp = initialize_exp()
device_setup = exp.create_device_setup()

exp_signals = exp.signals(qubit)
signal_map_default = exp.signal_map_default(qubit)

# %% Experiment Parameters
simulate = False
n_average = 4000

seq_lengths = list(range(1, 200, 5))
drive_length = 40e-9
n_seq_per_length = 5
max_seq_duration = max(seq_lengths) * 3 * drive_length

# %% Experiment Definition
exp_rb = Experiment(
    uid="RandomizedBenchmark",
    signals=exp_signals

)

gate_map = {
    "I": pulses.pi_pulse(qubit),
    "X": pulses.pi_pulse(qubit),
    "Y": pulses.pi_pulse(qubit),
    "X/2": pulses.pi_pulse(qubit),
    "Y/2": pulses.pi_pulse(qubit),
    "-X/2": pulses.pi_pulse(qubit),
    "-Y/2": pulses.pi_pulse(qubit),
}

with exp_rb.acquire_loop_rt(
        uid=f"rb_shots",
        count=n_average,
        averaging_mode=AveragingMode.CYCLIC,
        acquisition_type=AcquisitionType.SPECTROSCOPY

):
    for seq_length in seq_lengths:
        for _ in range(n_seq_per_length):
            with exp_rb.section(
                    length=max_seq_duration, alignment=SectionAlignment.RIGHT
            ):
                generate_play_rb_pulses(
                    exp_rb, f"drive_{qubit}", seq_length, clifford_parametrized, gate_map
                )

            # readout and data acquisition
            with exp_rb.section():
                exp_rb.reserve(f"drive_{qubit}")
                exp_rb.play(signal=f"measure",
                            pulse=pulses.readout_pulse(qubit),
                            phase=qubit_parameters[qubit]['angle'])
                exp_rb.acquire(
                    signal=f"acquire",
                    handle=f"acq_{seq_length}",
                    length=qubit_parameters[qubit]["res_len"]
                )

            with exp_rb.section():
                exp_rb.reserve(f"drive_{qubit}")
                exp_rb.delay(signal=f"measure", time=120e-6)

# %% create experiment and pulses calibration
exp_rb.set_signal_map(signal_map_default)

# %% Compile, Generate Pulse Sheet, and Plot Simulated Signals

session = Session(device_setup=device_setup)
session.connect(do_emulation=False)
compiler_settings = {"SHFSG_MIN_PLAYWAVE_HINT": 256, "SHFSG_MIN_PLAYZERO_HINT": 512}
compiled_exp_rb = session.compile(exp_rb, compiler_settings=compiler_settings)

if simulate:
    plot_simulation(compiled_exp_rb, start_time=0, length=1e-3)

# %% get results
my_results = session.run(compiled_exp_rb)


# %% fit
def power_law(power, a, b, p):
    return a * (p ** power) + b


avg_meas = []
for seq_length in seq_lengths:
    avg_meas.append(my_results.get_data(f"acq_{seq_length}"))

# %% plot
amplitude = np.real(avg_meas)
amplitude = np.mean(amplitude, axis=1)
amplitude = 1 - correct_axis(amplitude, qubit_parameters[qubit]["ge"])

pars = curve_fit(
    f=power_law,
    xdata=seq_lengths,
    ydata=amplitude,
    p0=[0.5, 0.5, 0.9],
    bounds=(-np.inf, np.inf),
    maxfev=2000,
)[0]

plt.title(f'RB {qubit}', fontsize=18)

x_dense = np.linspace(0, seq_lengths[-1], 1000, )
plt.plot(x_dense, power_law(x_dense, *pars), label="fit")
plt.plot(seq_lengths, amplitude, ".", label='data')

rc = (1 - pars[2]) / 2
rg = rc / 2
print(f"rc = {rc:e}")
print(f"rg = = {rg:e}")
print(f"1 - rg = {1 - rg}")
plt.xlabel("Sequence Length")
plt.ylabel("Average Fidelity")
plt.legend()
plt.show()

# %% save

data = {
    'x_data': seq_lengths,
    'y_data': amplitude
}

meta_data = {}

exp_data = ExperimentData(meta_data=meta_data, data=data)
exp_data.save_data(f'rb_{qubit}')

from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
import numpy as np
import matplotlib.pyplot as plt
from laboneq.simple import *
from helper import pulses, exp_helper
from qubit_parameters import qubit_parameters
from helper.utility_functions import cos_wave, correct_axis

# %% parameters
exp = exp_helper.initialize_exp()
device_setup = exp.create_device_setup('software')
# exp_signals = exp.signals()
# signal_map_default = exp.signal_map_default()

# %%
session = Session(device_setup=device_setup)
session.connect(do_emulation=True)
# time.sleep(10)

# %% amplitude sweep
simulate = True

amp_min = 0  # 0.5*8e-3 # [V]
amp_max = 1 / 2  # 0.8 # [V]
amp_num = 100

exp_repetitions = 1000
pis = 1

qubits = ['q1', 'q4']


def create_rabi_amp_sweep(amp_num, uid="rabi_amp"):
    return LinearSweepParameter(uid=uid, start=amp_min, stop=amp_max, count=amp_num)


amplitude_sweep1 = LinearSweepParameter(uid='axis1', start=0, stop=10 / 10, count=100)
amplitude_sweep2 = LinearSweepParameter(uid='axis2', start=1 / 3, stop=1 / 2, count=100)


# %%
def rabi_pulses(
        exp,
        qubit,
        sweep_parameter,
):
    # qubit excitation - pulse amplitude will be swept
    with exp.section():
        exp.play(signal=f"drive_{qubit}", pulse=pulses.many_pi_pulse(qubit, pis), amplitude=sweep_parameter)
    # qubit readout pulse and data acquisition
    with exp.section(trigger={f"measure_{qubit}": {"state": True}}):
        exp.reserve(signal=f"drive_{qubit}")
        # play readout pu
        exp.play(signal=f"measure_{qubit}", pulse=pulses.readout_pulse(qubit))
        # signal data acquisition
        exp.acquire(
            signal=f"acquire_{qubit}",
            handle=f'rabi_{qubit}',
            kernel=pulses.readout_pulse(qubit)

        )
    with exp.section():
        exp.delay(signal=f"measure_{qubit}", time=1e-6)


# %% Experiment Definition


def amplitude_rabi():
    exp_rabi = Experiment(
        uid="Amplitude Rabi",
        signals=[
            ExperimentSignal("measure_q1"),
            ExperimentSignal("acquire_q1"),
            ExperimentSignal("measure_q4"),
            ExperimentSignal("acquire_q4"),
            ExperimentSignal(f"drive_q1"),
            ExperimentSignal(f"drive_q4"),

        ]
    )
    with exp_rabi.acquire_loop_rt(
            uid="shots",
            count=exp_repetitions,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
            reset_oscillator_phase=True
    ):
        # inner loop - real-time sweep of qubit drive pulse amplitude
        with exp_rabi.sweep(uid="sweep", parameter=[amplitude_sweep1, amplitude_sweep2]):
            rabi_pulses(
                exp_rabi,
                qubits[0],
                sweep_parameter=amplitude_sweep1,
            )
            rabi_pulses(
                exp_rabi,
                qubits[1],
                sweep_parameter=amplitude_sweep2,
            )

    return exp_rabi


# %%
exp_rabi = amplitude_rabi()

signal_map_default = {"drive_q1": device_setup.logical_signal_groups[qubits[0]].logical_signals["drive_line"],
                      "measure_q1": device_setup.logical_signal_groups[qubits[0]].logical_signals["measure_line"],
                      "acquire_q1": device_setup.logical_signal_groups[qubits[0]].logical_signals["acquire_line"],
                      "drive_q4": device_setup.logical_signal_groups[qubits[1]].logical_signals["drive_line"],
                      "measure_q4": device_setup.logical_signal_groups[qubits[1]].logical_signals["measure_line"],
                      "acquire_q4": device_setup.logical_signal_groups[qubits[1]].logical_signals["acquire_line"]}

exp_rabi.set_signal_map(signal_map_default)

# %% compile and simulate
compiled_rabi = session.compile(exp_rabi)

if simulate:
    plot_simulation(compiled_rabi, start_time=0, length=15e-6,
                    signals=[f'drive_{qubits[0]}', f'drive_{qubits[1]}', f'measure_{qubits[0]}',
                             f'measure_{qubits[1]}', f'acquire_{qubits[0]}', f'acquire_{qubits[1]}'])

# %% Run, Save, and Plot results
rabi_results = session.run()

print(rabi_results.acquired_results)
# %% plot results KA
#
results_1 = rabi_results.get_data(f"rabi_{qubits[0]}")
results_2 = rabi_results.get_data(f"rabi_{qubits[1]}")

a = rabi_results.acquired_results['rabi_q1']
b = rabi_results.acquired_results['rabi_q4']
# print(abs(a.data))
# print(abs(b.data))

fig, axes = plt.subplots(2, 1, figsize=(8, 6))

amplitude_q1 = np.abs(results_1)
axes[0].plot(amplitude_sweep1.values, abs(a.data), color='blue', marker='o')
#
amplitude_q4 = np.abs(results_2)
axes[1].plot(amplitude_sweep2.values, abs(b.data), color='blue', marker='o')
#
plt.show()

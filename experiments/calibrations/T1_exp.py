from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from helper.pulses import *
from helper import pulses, exp_helper
from helper.utility_functions import correct_axis
from helper.experiment_results import ExperimentData
from helper.plotter import Plotter
from qubit_parameters import qubit_parameters
from helper.utility_functions import exp_decay


# %% parameters
qubit = "q1"
exp = exp_helper.initialize_exp()
device_setup = exp.create_device_setup()
exp_signals = exp.signals(qubit)
signal_map_default = exp.signal_map_default(qubit)

# %%
session = Session(device_setup=device_setup)
session.connect(do_emulation=False)

# %% Quantum system parameters
simulate = False
sweep_start = 0
sweep_stop = 100e-6
step_num = 100
ts = np.linspace(0, sweep_stop, step_num)
delay_sweep = SweepParameter(values=ts)
exp_repetitions = 1000

# %% Experiment Definition
def t1_exp():
    t1_exp = Experiment(
        uid="Amplitude Rabi",
        signals=exp_signals
    )
    with t1_exp.acquire_loop_rt(
            uid="rabi_shots",
            count=exp_repetitions,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
    ):
        # inner loop - real time sweep of Rabi ampitudes
        with t1_exp.sweep(uid="rabi_sweep", parameter=delay_sweep):
            # play qubit excitation pulse - pulse amplitude is swept
            with t1_exp.section(
                    uid="qubit_excitation"
            ):
                t1_exp.play(
                    signal=f"drive_{qubit}", pulse=pulses.pi_pulse(qubit),
                )
                t1_exp.delay(signal=f"measure", time=delay_sweep)
            # readout pulse and data acquisition
            with t1_exp.section(uid="readout_section", play_after="qubit_excitation"):
                # play readout pulse on measure line
                t1_exp.play(signal="measure",
                            pulse=pulses.readout_pulse(qubit),
                            phase=qubit_parameters[qubit]['angle'])
                # trigger signal data acquisition
                t1_exp.acquire(
                    signal="acquire",
                    handle="t1_handle",
                    length=qubit_parameters[qubit]["res_len"],
                )
            with t1_exp.section(uid="delay"):
                # relax time after readout - for qubit relaxation to groundstate and signal processing
                t1_exp.delay(signal="measure", time=120e-6)
    return t1_exp


# %% Create Experiment and Signal Map
t1_exp = t1_exp()
t1_exp.set_signal_map(signal_map_default)

# %%
compiled_rabi = session.compile(t1_exp)
if simulate:
    plot_simulation(compiled_rabi, start_time=0, length=10e-6, signals=[f'drive_{qubit}', 'measure', 'acquire'])

# %% Run, Save, and Plot results
t1_exp_results = session.run()

# %% plot parameters results KA
acquire_results = t1_exp_results.get_data("t1_handle")
amplitude = np.real(acquire_results)
amplitude = correct_axis(amplitude, qubit_parameters[qubit]["ge"])

# %% fitting
# Initial guess for the parameters
initial_guess = [1, 14.8e-6, 0.00018]

# Perform the curve fit
params, covariance = curve_fit(exp_decay, ts, amplitude, p0=initial_guess)

marker_size = 3
# Plot the original data and the fitted curve
plt.plot(ts * 1e6, amplitude, 'bo', label='Original Data', markersize=marker_size)
plt.plot(ts * 1e6, exp_decay(ts, *params), 'r-', label='Fitted Curve')
plt.xlabel('Time [mu]')
plt.ylabel('Amplitude')
plt.legend()
plt.title(f'T1 Experiment \nFitted T1 = {params[1] * 1e6:.2f} us')

plt.show()

print(f"Fitted T1 = {params[1] * 1e6} us")

# %% save results

meta_data = {
    'type': '1d',
    'plot_properties':
        {
            'x_label': 'Time [s]',
            'y_label': 'Amplitude',
            'title': 'T1 Experiment'
        },
    'experiment_properties':
        {
            'qubit': qubit,
            'repetitions': exp_repetitions,
        },
    'qubit_parameters': qubit_parameters,
}

data = {
    'x_data': ts,
    'y_data': amplitude
}

exp_data = ExperimentData(meta_data=meta_data, data=data)
exp_data.save_data(file_name=f't1_{qubit}')


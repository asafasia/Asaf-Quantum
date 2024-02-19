import matplotlib.pyplot as plt
from laboneq.simple import *
import numpy as np
from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
from helper import pulses, exp_helper
from helper.utility_functions import correct_axis, func_lorentz
from qubit_parameters import qubit_parameters
from helper.experiment_results import ExperimentData
from helper.plotter import Plotter

# %% parameters
qubit = "q1"
initializer = exp_helper.initialize_exp()
device_setup = initializer.create_device_setup()
exp_signals = initializer.signals(qubit)
signal_map_default = initializer.signal_map_default(qubit)

# %%
session = Session(device_setup=device_setup)
session.connect(do_emulation=False)

# %% exp parameters
simulate = False
exp_repetitions = 1000
amp = 1 / 1500
w0 = True

if w0:
    center = qubit_parameters[qubit]["qb_freq"]
else:
    center = qubit_parameters[qubit]["w125"]

span = 4e6
steps = 101

drive_LO = qubit_parameters[qubit]["qb_lo"]

dfs = np.linspace(start=center - span / 2, stop=center + span / 2, num=steps)  # for carrier calculation

freq_sweep = SweepParameter(uid=f'freq_sweep_{qubit}', values=dfs - drive_LO)  # sweep object


# %% experiment definition
def qubit_spectroscopy():
    exp_qspec = Experiment(
        uid="Qubit Spectroscopy",
        signals=exp_signals,
    )
    with exp_qspec.acquire_loop_rt(
            uid="freq_shots",
            count=exp_repetitions,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
    ):
        with exp_qspec.sweep(uid="qfreq_sweep", parameter=freq_sweep):
            with exp_qspec.section(uid="qubit_excitation"):
                exp_qspec.play(signal=f"drive_{qubit}", pulse=pulses.spec_pulse(qubit), amplitude=amp)

            with exp_qspec.section(
                    uid="readout_section", play_after="qubit_excitation", trigger={"measure": {"state": True}}
            ):
                exp_qspec.play(signal="measure",
                               pulse=pulses.readout_pulse(qubit),
                               phase=qubit_parameters[qubit]['angle'])
                exp_qspec.acquire(
                    signal="acquire",
                    handle="qubit_spec",
                    length=qubit_parameters[qubit]["res_len"],
                    # kernel = kernel
                )
            with exp_qspec.section(uid="delay"):
                # relax time after readout - for qubit relaxation to ground state and signal processing
                exp_qspec.delay(signal="measure", time=120e-6)

    return exp_qspec


# %% Run Exp
exp_calibration = Calibration()

exp_calibration[f"drive_{qubit}"] = SignalCalibration(
    oscillator=Oscillator(
        frequency=freq_sweep,
        modulation_type=ModulationType.HARDWARE,
    ),
)

exp_qspec = qubit_spectroscopy()
exp_qspec.set_calibration(exp_calibration)
exp_qspec.set_signal_map(signal_map_default)

# %% compile
compiled_qspec = session.compile(exp_qspec)
if simulate:
    plot_simulation(compiled_qspec, start_time=0, length=350e-6, signals=[f'drive_{qubit}', 'measure'])

qspec_results = session.run(compiled_qspec)

# %% plot results
results = qspec_results.get_data("qubit_spec")
amplitude = np.real(results)
amplitude = correct_axis(amplitude, qubit_parameters[qubit]["ge"])
phase_radians = np.unwrap(np.angle(results))

# guess = [1, 4.628e9 - 1000e6, -10, 0.0016]

# popt, pcov = opt.curve_fit(func_lorentz, dfs, amplitude, p0=guess)

resonanace_x = (qubit_parameters[qubit]["qb_freq"]) * 1e-6

anharmonicity = (qubit_parameters[qubit]["qb_freq"] - qubit_parameters[qubit]["w125"]) * 2 * 1e-9

drive_amp = qubit_parameters[qubit]['drive_amp'] * amp

flux_bias = qubit_parameters[qubit]['flux_bias']

new_min = dfs[np.argmax(amplitude)]

print(f"new min freq = {new_min * 1e-9} GHz")

if w0:
    plt.title(
        f'Qubit Spectroscopy {qubit} w0\n drive amp = {drive_amp:5f} V \n flux bias = {flux_bias:.5f} V \nresonanance = {center * 1e-6} MHz  ',
        fontsize=18)
else:
    plt.title(
        f'Qubit Spectroscopy {qubit} w1\n drive amp = {drive_amp:5f} V \n flux bias = {flux_bias:.5f} V \nresonanance = {center * 1e-6:.2f} MHz  ',
        fontsize=18)
plt.axvline(x=0, color='green', linestyle='--')
plt.ylim([-0.1, 1.1])
plt.plot(dfs * 1e-6 - center * 1e-6, amplitude, "k")
# plt.plot(dfs*1e-6 - center*1e-6,func_lorentz(dfs,*popt))
plt.xlabel('Detuning [MHz]')
plt.ylabel('Amplitude [a.u.]')
plt.show()
# print(f"width = {popt[0] * 1e-6} MHz")

# %% save results
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
    'qubit_parameters': qubit_parameters[qubit],

}

data = {
    'x_data': dfs,
    'y_data': amplitude,
}

exp_data = ExperimentData(meta_data=meta_data, data=data)
exp_data.save_data(file_name='qubit_spectroscopy')

# exp_plot = Plotter(meta_data=meta_data, data=data)
# exp_plot.plot()
# plt.show()

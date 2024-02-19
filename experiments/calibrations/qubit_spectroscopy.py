from laboneq.simple import *
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from laboneq.contrib.example_helpers.plotting.plot_helpers import (
    plot_simulation,
)

from qubit_parameters import qubit_parameters, update_qp
from helper.kernels import kernels
from helper.exp_helper import *
from helper.pulses import *

import scipy.optimize as opt


# %% parameters
qubit = "q2"

mode = 'spec'
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

# %%

session = Session(device_setup=device_setup)
session.connect(do_emulation=False)

# %%
save_labber=True
update_flux = False
simulate = False
n_avg = 300
amp = 1 / 5  # ~0.4 (q3) for 2nd E level, 1/100 for 1st E level
w0 = False
plot_from_json = False
center_axis = True
ground_max = False
p = 1

if w0:
    center = qubit_parameters[qubit]["qb_freq"]
else:
    center = qubit_parameters[qubit]["w125"]

span = 300e6
steps = 201

drive_LO = qubit_parameters[qubit]["qb_lo"]

dfs = np.linspace(start=center - span / 2, stop=center + span / 2, num=steps)  # for carrier calculation

freq_sweep = SweepParameter(uid=f'freq_sweep_{qubit}', values=dfs - drive_LO)  # sweep object

alpha = (qubit_parameters[qubit]['qb_freq'] - qubit_parameters[qubit]['w125']) * 2

update_qp(qubit, 'anharmonicity', alpha)
print('updated unharmonicity ', alpha)


# %%


def qubit_spectroscopy(freq_sweep):
    exp_qspec = Experiment(
        uid="Qubit Spectroscopy",
        signals=exp_signals,
    )
    with exp_qspec.acquire_loop_rt(
            uid="freq_shots",
            count=n_avg,
            acquisition_type=acquisition_type,
    ):
        with exp_qspec.sweep(uid="qfreq_sweep", parameter=freq_sweep):
            with exp_qspec.section(uid="qubit_excitation"):
                exp_qspec.play(signal=f"drive_{qubit}", pulse=spec_pulse(qubit), amplitude=amp,
                               marker={"marker1": {"enable": True}})

            with exp_qspec.section(
                    uid="readout_section", play_after="qubit_excitation",
            ):
                # play readout pulse on measure line
                exp_qspec.play(signal="measure",
                               pulse=readout_pulse(qubit),
                               phase=qubit_parameters[qubit]['angle'],

                               )

                # trigger signal data acquisition
                exp_qspec.acquire(
                    signal="acquire",
                    handle="qubit_spec",
                    kernel=kernel,
                )
            # delay between consecutive experiment
            with exp_qspec.section(uid="delay"):
                # relax time after readout - for qubit relaxation to groundstate and signal processing
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

# define experiment with frequency sweep for qubit 0
exp_qspec = qubit_spectroscopy(freq_sweep)
# apply calibration and signal map for qubit 0
exp_qspec.set_calibration(exp_calibration)
exp_qspec.set_signal_map(signal_map_default)
# %%
# compile the experiment on the open instrument session
compiled_qspec = session.compile(exp_qspec)

# %%plots
if simulate:
    plot_simulation(compiled_qspec, start_time=0, length=350e-6)

qspec_results = session.run(compiled_qspec)

# %% results

results = qspec_results.get_data("qubit_spec")

# %% set plotting variables

amplitude = np.abs(results)
# amplitude = correct_axis(amplitude,qubit_parameters[qubit]["ge"])

phase_radians = np.unwrap(np.angle(results))


# plots


def func_lorentz(x, width, pos, amp, off):
    return off + amp * width / (width ** 2 + (x - pos) ** 2)


guess = [1, 4.628e9 - 1000e6, -10, 0.0016]

popt, pcov = opt.curve_fit(func_lorentz, dfs, amplitude, p0=guess)

drive_amp = qubit_parameters[qubit]['drive_amp'] * amp

flux_bias = qubit_parameters[qubit]['flux_bias']

if ground_max:
    max_freq = dfs[np.argmax(amplitude)]
else:
    max_freq = dfs[np.argmin(amplitude)]

if w0:
    plt.title(
        f'Qubit Spectroscopy {qubit} w0\n drive amp = {amp:5f} V \n flux bias = {flux_bias:.5f} V \ncenter = {center * 1e-6} MHz  ',
        fontsize=18)
else:
    plt.title(
        f'Qubit Spectroscopy {qubit} w1\n drive amp = {amp:5f} V \n flux bias = {flux_bias:.5f} V \ncenter = {center * 1e-6:.2f} MHz  ',
        fontsize=18)

# plt.ylim([-0.2,1.2])
decimal_places = 2
plt.gca().xaxis.set_major_formatter(StrMethodFormatter(f"{{x:,.{decimal_places}f}}"))
# detuning plot
# plt.plot((dfs - center)*1e-6, amplitude, "k")
if center_axis:
    plt.plot((dfs - center) * 1e-6, amplitude, "k")
    plt.axvline(x=0, color='green', linestyle='--', label='current')
    plt.axvline(x=(max_freq - center) * 1e-6, color='blue', linestyle='--', label='new')

else:
    plt.plot(dfs * 1e-6, amplitude, "k")
    plt.axvline(x=center * 1e-6, color='green', linestyle='--', label='current')
    plt.axvline(x=max_freq * 1e-6, color='blue', linestyle='--', label='new')

# plt.ylim([0,1])


# plt.plot(dfs*1e-6 - center*1e-6,func_lorentz(dfs,*popt))
plt.xlabel('Detuning [MHz]')
plt.ylabel('Amplitude [a.u.]')
plt.legend()
plt.show()


# %%


def inv_parabola(frequency, qubit):
    a, b, c = qubit_parameters[qubit]['abc']

    f1 = (-b + np.sqrt(b ** 2 - 4 * a * (c - frequency))) / (2 * a)
    f2 = (-b - np.sqrt(b ** 2 - 4 * a * (c - frequency))) / (2 * a)

    if abs(f1) < abs(f2):
        return f1
    else:
        return f2

    return min(abs(f1), abs(f2))


detuning = max_freq - center

new_flux_bias = flux_bias * (1 + detuning * 1e-6 / 100 * p)

if update_flux and w0 == True:
    print('old_flux_point = :', flux_bias)

    print('new_flux_point = :', new_flux_bias)
    user_input = input("Do you want to update the ***flux***? [y/n]")

    if user_input == 'y':
        update_qp(qubit, 'flux_bias', new_flux_bias)
        print('updated flux!!!')

    elif user_input == 'n':
        print('not_updated')
    else:
        raise Exception("Invalid input")
else:
    user_input = input("Do you want to update the ***frequency***? [y/n]")

    if w0:
        update_string = 'qb_freq'
    else:
        update_string = 'w125'

    if user_input == 'y':
        update_qp(qubit, update_string, max_freq)
        anharmonicity = 2 * abs(qubit_parameters[qubit]['w125'] - qubit_parameters[qubit]['qb_freq'])
        update_qp(qubit, 'anharmonicity', anharmonicity)

        print('updated freq!!!')

    elif user_input == 'n':
        print('not_updated')
    else:
        raise Exception("Invalid input")

if save_labber:
    import labber.labber_util as lu

    measured_data = dict(amplitude=np.array(results))
    sweep_parameters = dict(frequency=np.array(dfs))
    units = dict(frequency='Hz')
    meta_data = dict(user="Guy", tags=[qubit, 'spectroscopy'], qubit_parameters=qubit_parameters)
    exp_result = dict(measured_data=measured_data, sweep_parameters=sweep_parameters, units=units,
                      meta_data=meta_data)

    lu.create_logfile("qubit_spectroscopy", **exp_result, loop_type="1d")


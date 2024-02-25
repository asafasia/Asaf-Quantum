import matplotlib.pyplot as plt
from helper.kernels import kernels
from helper.exp_helper import *
from pulses import *
from qubit_parameters import qubit_parameters, update_qp
from helper.utility_functions import iq_helper, mode

# %% devise setup
qubit = "q1"
mode_type = 'disc'
modulation_type, acquisition_type, kernel = mode(qubit, mode_type)

# %%
exp = initialize_exp()
device_setup = exp.create_device_setup(modulation_type)
exp_signals = exp.signals(qubit)
signal_map_default = exp.signal_map_default(qubit)

# %% parameters
do_emulation = False
simulate = True
exp_repetitions = 10000
show_clouds = True

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
        count=exp_repetitions,
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
        count=exp_repetitions,
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
raw_0 = results_0.get_data("ac_0")
raw_1 = results_1.get_data("ac_1")

ground, excited, threshold, angle = iq_helper([raw_0, raw_1])

if mode_type == 'int':
    plt.plot(ground.real, ground.imag, '.', alpha=0.6, label='ground', )
    plt.plot(excited.real, excited.imag, '.', alpha=0.6, label='excited')
    plt.axvline(x=threshold, color='b', linestyle='--')
    plt.plot(np.mean(ground).real, np.mean(ground).imag, 'x', color='black')
    plt.plot(np.mean(excited).real, np.mean(excited).imag, 'x', color='black')
    plt.xlabel('I [a.u.]')
    plt.ylabel('Q [a.u.]')
    plt.title(f"IQ cloud {qubit}")
    plt.legend()
    plt.show()

if mode_type == 'int':
    bins = 80
else:
    bins = 8

plt.hist(raw_0.real, bins=bins, edgecolor='black', label='ground state', alpha=0.6, )
plt.hist(raw_1.real, bins=bins, edgecolor='black', label='excited state', alpha=0.5)
plt.title(f" Histogram {qubit} ")
plt.xlabel('I [a.u.]')
plt.ylabel('number')
plt.legend()
plt.show()

if mode_type == 'int':
    print(f"threshold = {threshold}")
    print(f"angle = {angle}")
    print("updated qubit parameters!")
    update_qp(qubit, 'threshold', threshold)
    update_qp(qubit, 'angle', angle + qubit_parameters[qubit]['angle'])
else:
    print(f'ground:  {np.mean(raw_0.real)}')
    print(f'excited:  {np.mean(1 - raw_1.real)}')

    print(f'estimated readout fidelity: {abs((np.mean(raw_0) + 1 - np.mean(raw_1)) / 2):.2f}')

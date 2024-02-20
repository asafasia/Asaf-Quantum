import matplotlib.pyplot as plt
from helper import project_path
from helper.exp_helper import *
from pulses import *
from qubit_parameters import *
from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation

# %% devise setup
qubit = "q3"

exp = initialize_exp()
device_setup = exp.create_device_setup(modulation_type='software')
exp_signals = exp.signals(qubit)
signal_map_default = exp.signal_map_default(qubit)

# %% 
session = Session(device_setup=device_setup)
session.connect(do_emulation=False)

# %% parameters
simulate = False
show_clouds = False
exp_repetitions = 32000

# %%
exp_0 = Experiment(
    uid="Optimal weights",
    signals=exp_signals
)

with exp_0.acquire_loop_rt(
        uid="shots",
        count=exp_repetitions,
        averaging_mode=AveragingMode.CYCLIC,
        acquisition_type=AcquisitionType.RAW,
):
    with exp_0.section():
        exp_0.play(signal=f"drive_{qubit}", pulse=pi_pulse(qubit), amplitude=0)

    with exp_0.section():
        exp_0.play(signal="measure", pulse=readout_pulse(qubit), phase=qubit_parameters[qubit]['angle'])
        exp_0.acquire(
            signal="acquire", handle="ac_0", length=kernel_pulse(qubit).length
        )

    with exp_0.section():
        exp_0.delay(signal="measure", time=120e-6)

exp_1 = Experiment(
    uid="Optimal weights",
    signals=exp_signals

)

with exp_1.acquire_loop_rt(
        uid="shots",
        count=exp_repetitions,  # pow(2, average_exponent),
        averaging_mode=AveragingMode.CYCLIC,
        acquisition_type=AcquisitionType.RAW,
):
    with exp_1.section():
        exp_1.play(signal=f"drive_{qubit}", pulse=pi_pulse(qubit), amplitude=1)

    with exp_1.section():
        exp_1.reserve(signal=f"drive_{qubit}")
        exp_1.play(signal="measure",
                   pulse=readout_pulse(qubit),
                   phase=qubit_parameters[qubit]['angle']
                   )

        exp_1.acquire(
            signal="acquire", handle="ac_1", length=kernel_pulse(qubit).length
        )

    with exp_1.section():
        exp_1.delay(signal="measure", time=120e-6)

exp_0.set_signal_map(signal_map_default)
exp_1.set_signal_map(signal_map_default)

# %%
if simulate:
    plot_simulation(session.compile(exp_1), 1e-6, 01e-8, signals=[f"drive_{qubit}", "measure", "acquire"])

# %%
session = Session(device_setup=device_setup)
session.connect(do_emulation=False)

results_0 = session.run(exp_0)
raw_0 = results_0.get_data("ac_0")

results_1 = session.run(exp_1)
raw_1 = results_1.get_data("ac_1")

traces = [raw_0, raw_1]

# %% run the first experiment and access the data
print(f"saved_kernel_{qubit}")
np.savetxt(f"{project_path}/helper/kernels/traces_{qubit}.txt", traces, fmt='%s')

# %%

m0 = np.mean(np.abs(raw_0))
m1 = np.mean(np.abs(raw_1))

time = np.linspace(0, len(raw_0) / 2, len(raw_0))
plt.plot(time, np.abs(raw_0), alpha=0.5, label='ground')
plt.plot(time, np.abs(raw_1), alpha=0.5, label='excited')
plt.axhline(y=m0)
plt.axhline(y=m1)

plt.xlabel("Time (ns)")
plt.ylabel("Amplitude (a.u.)")
plt.legend()
plt.show()

# %%

plt.title(f"raw iq {qubit}")
plt.plot(raw_0.real, raw_0.imag, '.', alpha=0.5)
plt.plot(raw_1.real, raw_1.imag, '.', alpha=0.5)
plt.show()

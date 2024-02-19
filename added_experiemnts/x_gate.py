import matplotlib.pyplot as plt
from laboneq.contrib.example_helpers.plotting.plot_helpers import (
    plot_simulation,
)
from scipy.optimize import curve_fit

from helper.exp_helper import *
from helper.kernels import kernels
from helper.pulses import *
from qubit_parameters import qubit_parameters

# %% devise setup
qubit = "q5"

mode = 'int'
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
device_setup = exp.create_device_setup(modulation_type=modulation_type)
signal_map_default = exp.signal_map_default(qubit)
exp_signals = exp.signals(qubit)
# %%
session = Session(device_setup=device_setup)
session.connect(do_emulation=False)

# %% Experiment Parameters
simulate = False
points = 15
exp_repetitions = 1000


# %% Experiment Definition


def x_gate_rep():
    exp_rabi = Experiment(
        uid="Amplitude Rabi",
        signals=exp_signals
    )

    with exp_rabi.acquire_loop_rt(
            uid="rabi_shots",
            count=exp_repetitions,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=acquisition_type):

        for i in range(points):
            with exp_rabi.section(alignment=SectionAlignment.RIGHT):
                for i in range(i):
                    exp_rabi.play(signal=f"drive_{qubit}",
                                  pulse=pi_pulse(qubit), amplitude=1 / 2, )

            with exp_rabi.section():
                exp_rabi.reserve(f"drive_{qubit}")

                exp_rabi.play(signal="measure",
                              pulse=readout_pulse(qubit),
                              phase=0
                              )

                exp_rabi.acquire(
                    signal="acquire",
                    handle="amp_rabi",
                    kernel=kernel,

                )
            with exp_rabi.section(uid="delay"):
                exp_rabi.reserve(f"drive_{qubit}")

                exp_rabi.delay(signal="measure", time=120e-6)

    return exp_rabi


# %% create experiment

x_gate_rep = x_gate_rep()

x_gate_rep.set_signal_map(signal_map_default)

# %% Compile
compiler_settings = {"SHFSG_MIN_PLAYWAVE_HINT": 128, "SHFSG_MIN_PLAYZERO_HINT": 256}

compiled_rabi = session.compile(x_gate_rep, compiler_settings=compiler_settings)

if simulate:
    plot_simulation(compiled_rabi, start_time=0, length=1e-3)

# %%
rabi_results = session.run()

# %% plot

acquire_results = rabi_results.get_data("amp_rabi")
amplitude = acquire_results.real
amplitude_I = acquire_results.imag
amplitude_half = np.abs(acquire_results[1:-1:2])
amplidute_for_fit = np.abs(acquire_results[1:-1:4])
# amplitude = correct_axis(amplitude,qubit_parameters[qubit]["ge"])
amp_mean = np.mean(amplitude)
pi_amp = qubit_parameters[qubit]['pi_amp']

x = range(points)
x_half = x[1:-1:2]
x_for_fit = x[1:-1:4]

plt.title(f'X Gate Repetitions Experimnet {qubit}')
plt.title('NOT QASM, X gate, PHASE = pi/2')
# plt.axhline(y=0.5,color = 'black')
# plt.plot(range(points), amplitude, '.')
plt.plot(range(points), amplitude, label='Real')
plt.plot(range(points), amplitude_I, label='Imag')
plt.legend()


# plt.plot(x_half, amplitude_half, '.', color='green')
# plt.plot(x_half, amplitude_half)
#
# plt.plot(x_for_fit, amplidute_for_fit, '.', color='green')
# plt.plot(x_for_fit, amplidute_for_fit)


def sin(x, amplitude, T, phase, offset):
    return amplitude * np.sin(2 * np.pi / (T * 2) * x + np.pi) + 1 / 2


try:
    args = curve_fit(sin, x_for_fit, amplidute_for_fit, p0=[0.3, 100, 0, 0.5])[0]

    T = abs(args[1])
    print("T = ", T)

    new_amplitude = qubit_parameters[qubit]['pi_amp'] * (1 - 2 / T)

    # plt.plot(x_for_fit, sin(x_for_fit, *args), color='red')
    print("old amplitude = ", qubit_parameters[qubit]['pi_amp'])
    print("new amplitude = ", new_amplitude)
except:
    print("fit failed")

plt.xlabel('Number of X/2 Gates')
plt.ylabel('Population')
plt.show()

from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from helper.kernels import kernels
from helper.exp_helper import *
from helper.pulses import *
from qubit_parameters import qubit_parameters, update_qp

# %% parameters
qubit = "q2"

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

# %% 
session = Session(device_setup=device_setup)
session.connect(do_emulation=False)

# %% amplitude sweep
simulate = False
steps = 300
pis = 4
n_avg = 1000
plot_from_json = False
sweep_rabi_amp = LinearSweepParameter(start=0, stop=1, count=steps)

# %% Experiment Definition
def amplitude_rabi(amplitude_sweep):
    exp_rabi = Experiment(
        uid="Amplitude Rabi",
        signals=exp_signals
    )

    with exp_rabi.acquire_loop_rt(
            uid="rabi_shots",
            count=n_avg,
            acquisition_type=acquisition_type,
    ):
        with exp_rabi.sweep(uid="rabi_sweep", parameter=amplitude_sweep):
            with exp_rabi.section(
                    uid="qubit_excitation"
            ):
                exp_rabi.play(signal=f"drive_{qubit}",
                              pulse=many_pi_pulse(qubit, pis),
                              amplitude=amplitude_sweep
                              )

            with exp_rabi.section(uid="readout_section", play_after="qubit_excitation",
                                  trigger={"measure": {"state": True}}):
                exp_rabi.play(signal="measure",
                              pulse=readout_pulse(qubit),
                              phase=qubit_parameters[qubit]['angle'])

                exp_rabi.acquire(
                    signal="acquire",
                    handle="amp_rabi",
                    kernel=kernel
                )
            with exp_rabi.section(uid="delay"):
                exp_rabi.delay(signal="measure", time=120e-6)
    return exp_rabi


# %%
exp_rabi = amplitude_rabi(sweep_rabi_amp)
exp_rabi.set_signal_map(signal_map_default)

# %%
compiled_rabi = session.compile(exp_rabi)
if simulate:
    plot_simulation(compiled_rabi, start_time=0, length=15e-6)

# %% Run
rabi_results = session.run()

# %% aqcuire results
acquire_results = rabi_results.get_data("amp_rabi")
amplitude = np.abs(acquire_results)
rabi_amp = sweep_rabi_amp.values * qubit_parameters[qubit]["pi_amp"] * pis

def cos_wave(x, amplitude, T, phase, offset):
    return amplitude * np.cos(2 * np.pi / T * x + phase) + offset


guess = [(max(amplitude) - min(amplitude)) / 2, qubit_parameters[qubit]['pi_amp'] * 2, 0, np.mean(amplitude)]

params, params_covariance = curve_fit(cos_wave, rabi_amp, amplitude, p0=guess)

print("new amplitude = ", abs(params[1]) / 2)

plt.plot(rabi_amp, amplitude, '-', color='black', label='data')
plt.plot(rabi_amp, cos_wave(rabi_amp, *params),
         label=f'Fitted function, rabi Amplitude = {abs(params[1]) / 2:.4f}', color='red')
plt.xlabel('Drive Amplitude [a.u.]')
plt.ylabel('Amplitude [a.u.]')
plt.legend()
plt.title(f'Amplitude Rabi {qubit}', fontsize=18)

plt.tight_layout()
plt.show()

# %% update


user_input = input("Do you want to update the pi amplitude? [y/n]")
if user_input == 'y':
    update_qp(qubit, 'pi_amp', abs(params[1]) / 2)
    print('updated !!!')

elif user_input == 'n':
    print('not_updated')
else:
    raise Exception("Invalid input")

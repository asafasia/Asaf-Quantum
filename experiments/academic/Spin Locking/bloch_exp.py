from pprint import pprint
from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
import numpy as np
import matplotlib.pyplot as plt
from laboneq.simple import *
from helper import pulses, exp_helper
from helper.utility_functions import correct_axis
from qubit_parameters import qubit_parameters
import qutip

# %% parameters
qubit = "q4"
experiment = exp_helper.initialize_exp()
device_setup = experiment.create_device_setup()
exp_signals = experiment.signals(qubit)
signal_map_default = experiment.signal_map_default(qubit)

# %% 
session = Session(device_setup=device_setup)
session.connect(do_emulation=False)

# %% amplitude sweep
simulate = True
amp_num = 100
exp_repetitions = 1000
amplitude_sweep = LinearSweepParameter(start=0, stop=1, count=amp_num)

# %% Experiment Definition
def create_bloch_exp():
    bases = ['I', 'X', 'Y']
    bloch_exp = Experiment(
        uid="Bloch_experiment",
        signals=exp_signals
    )

    with bloch_exp.acquire_loop_rt(
            uid="rabi_shots",
            count=exp_repetitions,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
    ):

        for basis in bases:
            with bloch_exp.sweep(uid=f"sweep_basis{basis}", parameter=amplitude_sweep):
                with bloch_exp.section(uid=f'qubit_excitation_{basis}'):
                    bloch_exp.play(signal=f'drive_{qubit}', pulse=pulses.pi_pulse(qubit), amplitude=amplitude_sweep,
                                   length=100e-9)

                with bloch_exp.section(uid=f"change_basis_{basis}"):

                    if basis == 'I':
                        bloch_exp.play(signal=f'drive_{qubit}',
                                       pulse=pulses.pi_pulse(qubit),
                                       amplitude=0,
                                       phase=0
                                       )
                    elif basis == 'X':
                        bloch_exp.play(signal=f'drive_{qubit}',
                                       pulse=pulses.pi_pulse(qubit),
                                       amplitude=1/2,
                                       phase=0
                                       )
                    elif basis == 'Y':
                        bloch_exp.play(signal=f'drive_{qubit}',
                                       pulse=pulses.pi_pulse(qubit),
                                       amplitude=1/2,
                                       phase=np.pi / 2
                                       )

                with bloch_exp.section(uid=f"measure_{basis}", play_after=f"change_basis_{basis}"):

                    bloch_exp.play(signal="measure",
                                   pulse=pulses.readout_pulse(qubit),
                                   phase=qubit_parameters[qubit]['angle'])

                    bloch_exp.acquire(
                        signal="acquire",
                        handle=f"handle_{basis}",
                        length=qubit_parameters[qubit]["res_len"],
                        # kernel = kernels[qubit]
                    )
                with bloch_exp.section():
                    bloch_exp.delay(signal="measure", time=120e-6)

    bloch_exp.set_signal_map(signal_map_default)

    return bloch_exp


# %%
bloch_exp = create_bloch_exp()
compiled_bloch_exp = session.compile(bloch_exp)

if simulate:
    plot_simulation(compiled_bloch_exp, start_time=1e-6, length=10e-6, signals=[f'drive_{qubit}', 'measure'])

# %% Run
bloch_exp_results = session.run()

# %% plot results
acquire_results_I = bloch_exp_results.get_data("handle_I")
acquire_results_X = bloch_exp_results.get_data("handle_X")
acquire_results_Y = bloch_exp_results.get_data("handle_Y")

I = np.real(acquire_results_I)
X = np.real(acquire_results_X)
Y = np.real(acquire_results_Y)

I = correct_axis(I, qubit_parameters[qubit]["ge"])
X = correct_axis(X, qubit_parameters[qubit]["ge"])
Y = correct_axis(Y, qubit_parameters[qubit]["ge"])

plt.plot(amplitude_sweep.values, I)
plt.plot(amplitude_sweep.values, X)
plt.plot(amplitude_sweep.values, Y)

plt.xlabel('Drive Amplitude [a.u.]')
plt.ylabel('Amplitude [a.u.]')
plt.legend()
plt.title(f'Amplitude Rabi {qubit}', fontsize=18)

plt.show()

# %% bloch sphere

b = qutip.Bloch()
b.make_sphere()

pnts = [X*2 - 1, Y*2 - 1, I*2 - 1]
b.add_points(pnts)
b.render()
plt.show()



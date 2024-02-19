import matplotlib.pyplot as plt
import numpy as np
from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
from helper.pulses import *
from helper import pulses, exp_helper
from helper.experiment_results import ExperimentData
from helper.plotter import Plotter
from qubit_parameters import update_qp, qubit_parameters

# %% devise setup
qubit = "q4"
exp = exp_helper.initialize_exp()
device_setup = exp.create_device_setup(modulation_type='hardware')
exp_signals = exp.signals(qubit)
signal_map_default = exp.signal_map_default(qubit)

# %% parameters
do_emulation = False
simulate = True
show_clouds = True
exp_repetitions = 500
relax = 1e-6

amplitudes = SweepParameter(uid='asdad', values=np.linspace(0.1, 1, 4))


# %% experiment
def exp_singleshot():
    exp_singleshot = Experiment(
        uid="Single Shot",
        signals=[
            ExperimentSignal(f"drive_{qubit}"),
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )
    with exp_singleshot.sweep(uid='a', parameter=amplitudes):
        with exp_singleshot.acquire_loop_rt(count=exp_repetitions,
                                            averaging_mode=AveragingMode.SINGLE_SHOT,
                                            acquisition_type=AcquisitionType.INTEGRATION
                                            ):
            ### start with qubit in the ground state
            with exp_singleshot.section(uid="qubit_excitation_g",
                                        alignment=SectionAlignment.RIGHT):
                exp_singleshot.play(signal=f"drive_{qubit}", pulse=pulses.pi_pulse(qubit), amplitude=0.0)
            # readout pulse and data acquisition
            with exp_singleshot.section(uid="readout_section_g", play_after="qubit_excitation_g", length=2e-6):
                # play readout pulse on measure line
                exp_singleshot.play(signal="measure", pulse=pulses.readout_pulse(qubit), amplitude=amplitudes)
            #     # trigger signal data acquisition
            #     exp_singleshot.acquire(
            #         signal="acquire",
            #         handle="q0_ground",
            #         kernel=pulses.readout_pulse(qubit),
            #
            #     )
            # with exp_singleshot.section(uid="delay_g", length=relax):
            #     # relax time after readout - for qubit relaxation to groundstate and signal processing
            #     exp_singleshot.reserve(signal="measure")
            # #
            # ### play qubit excitation pulse
            # with exp_singleshot.section(uid="qubit_excitation_e", play_after="delay_g"):
            #     exp_singleshot.play(signal=f"drive_{qubit}", pulse=pulses.pi_pulse(qubit))
            # # readout pulse and data acquisition
            # with exp_singleshot.section(uid="readout_section_e", play_after="qubit_excitation_e", length=2e-6):
            #     # play readout pulse on measure line
            #     exp_singleshot.play(signal="measure", pulse=pulses.readout_pulse(qubit), amplitude=amplitudes)
            #     # trigger signal data acquisition
            #     exp_singleshot.acquire(
            #         signal="acquire",
            #         handle="q0_excited",
            #         kernel=pulses.readout_pulse(qubit),
            #
            #     )
            # with exp_singleshot.section(uid="delay_e", length=relax):
            #     # relax time after readout - for qubit relaxation to groundstate and signal processing
            #     exp_singleshot.reserve(signal="measure")

    return exp_singleshot


# %% run
session = Session(device_setup=device_setup)
session.connect(do_emulation=do_emulation)

exp_shot = exp_singleshot()
signal_map_default = {f"drive_{qubit}": device_setup.logical_signal_groups[qubit].logical_signals["drive_line"],
                      "measure": device_setup.logical_signal_groups[qubit].logical_signals["measure_line"],
                      "acquire": device_setup.logical_signal_groups[qubit].logical_signals["acquire_line"]}

exp_shot.set_signal_map(signal_map_default)

# %% compile
compiled_exp = session.compile(exp_shot)

if simulate:
    plot_simulation(compiled_exp, 0e-6, 3e-6, signals=["drive_q4_ef", "measure"])



my_results = session.run(compiled_exp)

amplitude_g = my_results.get_data("q0_ground")
amplitude_e = my_results.get_data("q0_excited")

m_g = np.mean(amplitude_g, axis=1)
m_e = np.mean(amplitude_e, axis=1)

v_g = np.var(amplitude_g, axis=1)
v_e = np.var(amplitude_e, axis=1)

sep_g = abs(2 * (m_g - m_e) / (np.sqrt(v_g) + np.sqrt(v_e)))

plt.plot(range(10), sep_g)
plt.show()

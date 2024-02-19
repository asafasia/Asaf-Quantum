from laboneq.contrib.example_helpers.plotting.plot_helpers import *
from helper import qubit_parameters, pulses, exp_helper
from helper.utility_functions import correct_axis
from laboneq.simple import *
import spin_locking_utils as ut

# %% experiment initialization

qubit = "q4"
initializer = exp_helper.initialize_exp()
device_setup = initializer.create_device_setup()

exp_signals = initializer.signals(qubit)
signal_map_default = initializer.signal_map_default(qubit)

# %% exp parameters

simulate = True
n_avg = 2

delay_step = 3
delay_stop = 1e-6

index_sweep = list(range(0, 41, 40))

print(index_sweep)

freq_sweep = SweepParameter(uid="freq_RF_sweep", values=index_sweep)

# %%
exp = Experiment(uid="test_experiment",
                 signals=exp_signals,
                 )

with exp.acquire_loop_rt(uid="RT_shots", count=n_avg):
    with exp.sweep(uid='freq_RF_sweep', parameter=freq_sweep):
        with exp.section(uid="ramp_up"):
            exp.play(signal=f"drive_{qubit}",
                     pulse=ut.ramp_pulse(qubit),
                     pulse_parameters={"freq": freq_sweep,
                                       'ramp_state': 'up'
                                       }
                     )
        #
        with exp.section(uid="delay"):
            exp.play(signal=f"drive_{qubit}",
                     pulse=ut.delay_pulse(qubit),
                     )

        with exp.section(uid="ramp_down"):
            exp.play(signal=f"drive_{qubit}",
                     pulse=ut.ramp_pulse(qubit),
                     pulse_parameters={"freq": freq_sweep,
                                       'ramp_state': 'down'
                                       }
                     )

        with exp.section(uid="readout_section", play_after="ramp_down"):
            # play readout pulse on measure line
            exp.play(signal="measure",
                     pulse=pulses.readout_pulse(qubit),
                     phase=qubit_parameters[qubit]["angle"]
                     )
            # trigger signal data acquisition
            exp.acquire(
                signal="acquire",
                handle="res",
                length=pulses.readout_pulse(qubit).length,
            )
        with exp.section(uid="relax"):
            # relax time after readout - for qubit relaxation to groundstate and signal processing
            exp.delay(signal="measure",
                      time=1e-6
                      )

exp.set_signal_map(signal_map_default)

# %%
session = Session(device_setup=device_setup)
session.connect(do_emulation=True)
compiled_exp = session.compile(exp)
if simulate:
    plot_simulation(compiled_exp, 0e-6, 20e-6, signals=['drive_q4', 'measure'])

import matplotlib.pyplot as plt
from fractions import Fraction
from laboneq.simple import *
from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
import numpy as np
from helper import pulses, exp_helper
from helper.utility_functions import correct_axis
from qubit_parameters import qubit_parameters
from helper.experiment_results import ExperimentData
from helper.plotter import Plotter

# %% parameters
qubit = "q4"
exp = exp_helper.initialize_exp()
device_setup = exp.create_device_setup()
signal_map_default = exp.signal_map_default(qubit)
exp_signals = exp.signals(qubit)

# %% session
session = Session(device_setup=device_setup)
session.connect(do_emulation=False)

# %% parameters
simulate = False
p_type = 'Square'  # square,gaussian

exp_repetitions = 1000

n_lorenz = Fraction(3, 5)
p = 0.0001

center = qubit_parameters[qubit]["qb_freq"]
freq_span = 200e6
freq_steps = 50
length = 1e-6
max_amp = 1/2
amp_steps = 20
half_pulse = False

dfs = np.linspace(start=center - freq_span / 2, stop=center + freq_span / 2, num=freq_steps)  # for carrier calculation
dAs = np.linspace(start=0, stop=1, num=amp_steps)

drive_LO = qubit_parameters[qubit]["qb_lo"]
freq_sweep = SweepParameter(uid=f'freq_sweep_{qubit}', values=dfs - drive_LO)  # sweep object
amplitude_sweep = SweepParameter(uid=f"amplitude_sweep_{qubit}", values=dAs)


# %% exp
def power_broadening():
    power_broadening_exp = Experiment(
        uid="Qubit Spectroscopy",
        signals=[
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
            ExperimentSignal(f"drive_{qubit}"),
        ],
    )
    with power_broadening_exp.acquire_loop_rt(
            uid="freq_shots",
            count=exp_repetitions,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
    ):
        with power_broadening_exp.sweep(uid="freq_RF_sweep", parameter=freq_sweep):
            with power_broadening_exp.sweep(uid="delay_sweep", parameter=amplitude_sweep):

                with power_broadening_exp.section(uid="qubit_excitation"):
                    power_broadening_exp.play(
                        signal=f"drive_{qubit}",
                        pulse=pulses.power_broadening_pulse(
                            qubit=qubit,
                            amplitude=max_amp,
                            length=length,
                            pulse_type=p_type,
                            p=p,
                            n=n_lorenz

                        ),
                        amplitude=amplitude_sweep
                    )

                with power_broadening_exp.section(
                        uid="readout_section", play_after="qubit_excitation"
                ):
                    power_broadening_exp.play(signal="measure",
                                              pulse=pulses.readout_pulse(qubit),
                                              phase=qubit_parameters[qubit]['angle'])
                    power_broadening_exp.acquire(
                        signal="acquire",
                        handle="qubit_spec",
                        length=qubit_parameters[qubit]["res_len"],

                    )
                with power_broadening_exp.section(uid="delay"):
                    power_broadening_exp.delay(signal="measure", time=120e-6)

    return power_broadening_exp


# %% Run Exp
exp_calibration = Calibration()

exp_calibration[f"drive_{qubit}"] = SignalCalibration(
    oscillator=Oscillator(
        frequency=freq_sweep,
        modulation_type=ModulationType.HARDWARE,
    ),

)

bp_exp = power_broadening()
bp_exp.set_calibration(exp_calibration)

signal_map_default = {

    "drive_q4": device_setup.logical_signal_groups[qubit].logical_signals["drive_line"],
    "measure": device_setup.logical_signal_groups[qubit].logical_signals[
        "measure_line"
    ],
    "acquire": device_setup.logical_signal_groups[qubit].logical_signals[
        "acquire_line"
    ],
}

bp_exp.set_signal_map(signal_map_default)

# %% simulate
if simulate:
    compiled_bp_exp = session.compile(bp_exp)
    plot_simulation(compiled_bp_exp, start_time=0, length=7e-6, signals=[f'drive_{qubit}', 'measure', 'acquire', ])

# %% run
compiled_bp_exp = session.run(bp_exp)

# %% plot results
acquire_results = compiled_bp_exp.get_data("qubit_spec")
amplitude = acquire_results.real
amplitude = correct_axis(amplitude, qubit_parameters[qubit]["ge"])
amplitude = amplitude.T
x, y = np.meshgrid(dfs - qubit_parameters[qubit]["qb_freq"], dAs * max_amp)

plt.pcolormesh(x*1e-6 , y /qubit_parameters[qubit]['pi_amp'], amplitude)
plt.xlabel('Detuning [MHz]')
plt.ylabel("Ampiltude [Rabi amplitude unit]", )
plt.colorbar()
plt.show()

# %% save
meta_data = {
    'type': '2d',
    'plot_properties': None,
    'experiment_properties':
        {
            'qubit': qubit,
            'pulse_type': p_type,
            'pulse_length': length,
            'max_amp': max_amp,
            'half_pulse': half_pulse,
            'n_lorenz': str(n_lorenz),
            'pulse_cutting': p,
            'repetitions': exp_repetitions,
            'qubit_parameters': qubit_parameters,
        }
}

data = {
    'x_data': dfs,
    'y_data': dAs,
    'z_data': amplitude,
}

exp_data = ExperimentData(meta_data=meta_data, data=data)
exp_data.save_data(file_name='power_broadening')

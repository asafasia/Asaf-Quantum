import matplotlib.pyplot as plt
from fractions import Fraction
from laboneq.simple import *
from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
from helper import pulses, exp_helper
from helper.utility_functions import correct_axis
from qubit_parameters import qubit_parameters
from helper.experiment_results import ExperimentData
from pb_utils import *
from helper.kernels import kernels
import labber.labber_util as lu

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

kernel = pulses.readout_pulse(qubit) if mode == 'spec' else kernels[qubit]

exp = exp_helper.initialize_exp()
device_setup = exp.create_device_setup(modulation_type)
signal_map_default = exp.signal_map_default(qubit)
exp_signals = exp.signals(qubit)

# %% session
session = Session(device_setup=device_setup)
session.connect(do_emulation=False)

# %% parameters
simulate = False
p_type = 'Lorentzian'  # Square,Gaussian,Lorentzian

exp_repetitions = 500

n_lorenz = Fraction(1, 2)
p = 0.0001

center = qubit_parameters[qubit]["qb_freq"]
freq_span = 30e6
freq_steps = 30
length = 20e-6
max_amp = 1
amp_steps = 10
half_pulse = False
rotate_iq = True

dfs = np.linspace(start=center - freq_span / 2, stop=center + freq_span / 2, num=freq_steps)  # for carrier calculation
dAs = np.linspace(start=0, stop=1, num=amp_steps)

drive_LO = qubit_parameters[qubit]["qb_lo"]
freq_sweep = SweepParameter(uid=f'freq_sweep_{qubit}', values=dfs - drive_LO)  # sweep object
amplitude_sweep = SweepParameter(uid=f"amplitude_sweep_{qubit}", values=dAs)


# %% exp
def power_broadening():
    power_broadening_exp = Experiment(
        uid="Qubit Spectroscopy",
        signals=exp_signals,
    )
    with power_broadening_exp.acquire_loop_rt(
            uid="freq_shots",
            count=exp_repetitions,
            acquisition_type=acquisition_type,
    ):
        with power_broadening_exp.sweep(uid="freq_RF_sweep", parameter=freq_sweep):
            with power_broadening_exp.sweep(uid="delay_sweep", parameter=amplitude_sweep):
                with power_broadening_exp.section(uid="qubit_excitation"):
                    power_broadening_exp.play(
                        signal=f"drive_{qubit}",
                        pulse=power_broadening_pulse(
                            qubit=qubit,
                            amplitude=max_amp,
                            length=length,
                            pulse_type=p_type,
                            p=p,
                            n=n_lorenz,
                            half_pulse = half_pulse

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
                        kernel=kernel,

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

bp_exp.set_signal_map(signal_map_default)

# %% simulate
if simulate:
    compiled_bp_exp = session.compile(bp_exp)
    plot_simulation(compiled_bp_exp, start_time=0, length=7e-6, signals=[f'drive_{qubit}', 'measure', 'acquire', ])

# %% run
compiled_bp_exp = session.run(bp_exp)

# %% plot results
acquire_results = compiled_bp_exp.get_data("qubit_spec")
amplitude = abs(acquire_results)
amplitude = amplitude.T
if rotate_iq:
    amplitude = 1 - amplitude

x, y = np.meshgrid(dfs - qubit_parameters[qubit]["qb_freq"], dAs * max_amp)

plt.pcolormesh(x * 1e-6, y / qubit_parameters[qubit]['pi_amp'], amplitude)
plt.xlabel('Detuning [MHz]')
plt.ylabel("Ampiltude [Rabi amplitude unit]", )
plt.colorbar()
plt.show()

# %% save_to_labber
measured_data = {}
measured_data['amplitude'] = amplitude

sweep_parameters = dict(amplitude=dfs - qubit_parameters[qubit]["qb_freq"],
                        detuning=dAs * max_amp)
units = dict(amplitude="a.u", detuning="Hz")

meta_data = dict(tags=["Nadav-Lab", "Power Broadening"],
                 user="Asaf",
                 qubit=qubit,
                 qubit_parameters=qubit_parameters)

exp_result = dict(measured_data=measured_data,
                  sweep_parameters=sweep_parameters,
                  units=units,
                  meta_data=meta_data)

lu.create_logfile("power_broadening", **exp_result, loop_type="2d")

import numpy as np
from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
from scipy.optimize import curve_fit
from helper import pulses, exp_helper
from laboneq.simple import *
import spin_locking_utils as ut
from qubit_parameters import qubit_parameters
from helper.experiment_results import ExperimentData
from matplotlib import pyplot as plt
from helper.utility_functions import correct_axis, cos_wave_exp
from helper.kernels import kernels

from helper.exp_helper import *
from helper.pulses import *

# %% experiment initialization
qubit = "q5"

mode = 'disc'
modulation_type = 'hardware' if mode == 'spec' else 'software'
if mode == 'spec':
    acquisition_type = AcquisitionType.SPECTROSCOPY
elif mode == 'int':
    acquisition_type = AcquisitionType.INTEGRATION
elif mode == 'disc':
    acquisition_type = AcquisitionType.DISCRIMINATION

kernel = pulses.readout_pulse(qubit) if mode == 'spec' else kernels[qubit]


initializer = exp_helper.initialize_exp()
device_setup = initializer.create_device_setup(modulation_type=modulation_type)

exp_signals = initializer.signals(qubit)
signal_map_default = initializer.signal_map_default(qubit)

# %% exp parameters
ramp_length = 200e-9
qubit_relax = 200e-6
simulate = False
exp_repetitions = 500

f_rabi = 70e6

amp = 2 * qubit_parameters[qubit]["pi_amp"] * qubit_parameters[qubit]["pi_len"] * f_rabi

delay_step = 100
delay_stop = 50e-6

delay_sweep = LinearSweepParameter(start=4e-9, stop=delay_stop, count=delay_step)

detunings = ut.get_detunings()[1:-1:25]

freq_sweep = SweepParameter(uid="freq_RF_sweep", values=detunings)


# %% experiment definition
def spin_lock(exp, qubit: str, basis: str = "Z"):
    with exp.section(uid=f"preparation_{basis}"):
        exp.play(
            signal=f"drive_{qubit}",
            pulse=pulses.pi_pulse(qubit),
            amplitude=1 / 2,
            length=40e-9,
        )
    #
    with exp.section(uid=f"up_{basis}", play_after=f"preparation_{basis}"):
        exp.delay(signal=f"drive_{qubit}_ef", time=40e-9)
        exp.play(
            signal=f"drive_{qubit}_ef",
            pulse=ut.ramp_pulse(qubit, amplitude=amp),
            pulse_parameters={"detuning": freq_sweep, "ramp_state": "up"},
            length=ramp_length,
        )
    #
    with exp.section(uid=f"delay_{basis}", play_after=f"up_{basis}"):
        exp.play(
            signal=f"drive_{qubit}_ef",
            pulse=ut.delay_pulse(qubit, amplitude=amp),
            length=delay_sweep,
        )
    #
    with exp.section(uid=f"down_{basis}"):
        exp.play(
            signal=f"drive_{qubit}_ef",
            pulse=ut.ramp_pulse(qubit, amplitude=amp),
            pulse_parameters={"detuning": freq_sweep, "ramp_state": "down"},
            length=ramp_length,
        )
        exp.delay(signal=f"drive_{qubit}_ef", time=10e-9)

    with exp.section(uid=f"change_basis_{basis}", play_after=f"down_{basis}"):
        if basis == "Z":
            exp.play(
                signal=f"drive_{qubit}",
                pulse=pulses.pi_pulse(qubit),
                amplitude=0,
                phase=-2*np.pi*freq_sweep*delay_sweep,
            )
        elif basis == "X":
            exp.play(
                signal=f"drive_{qubit}",
                pulse=pulses.pi_pulse(qubit),
                amplitude=1 / 2,
                phase=0,
            )
        elif basis == "Y":
            exp.play(
                signal=f"drive_{qubit}",
                pulse=pulses.pi_pulse(qubit),
                amplitude=1 / 2,
                phase=np.pi / 2,
            )



    with exp.section(
        uid=f"measure_{basis}",
        play_after=f"change_basis_{basis}",
        trigger={"measure": {"state": True}},
    ):
        exp.play(
            signal="measure",
            pulse=pulses.readout_pulse(qubit),
            phase=qubit_parameters[qubit]["angle"],
        )

        exp.acquire(
            signal="acquire",
            handle=f"handle_{basis}",
            kernel=kernel,
        )

    with exp.section(
        uid=f"relax_{basis}",
    ):
        exp.delay(
            signal="measure",
            time=qubit_relax,
        )


def create_spin_locking_exp():
    spin_locking_exp = Experiment(
        uid="Spin_locking_experiment",
        signals=[
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
            ExperimentSignal(f"drive_{qubit}_ef"),
            ExperimentSignal(f"drive_{qubit}"),
        ],
    )

    with spin_locking_exp.sweep(uid="freq_RF_sweep", parameter=freq_sweep):
        with spin_locking_exp.acquire_loop_rt(
            uid="RT_shots",
            count=exp_repetitions,
            acquisition_type=acquisition_type,
            reset_oscillator_phase=True,

        ):
            with spin_locking_exp.sweep(uid="rabi_sweep", parameter=delay_sweep):
                with spin_locking_exp.section(uid="x_basis"):
                    spin_lock(spin_locking_exp, qubit, basis="X")
                # with spin_locking_exp.section(uid="y_basis", play_after="x_basis"):
                #     spin_lock(spin_locking_exp, qubit, basis="Y")
                # with spin_locking_exp.section(uid="z_basis", play_after="y_basis"):
                #     spin_lock(spin_locking_exp, qubit, basis="Z")

    return spin_locking_exp


signal_map_default = {
    f"drive_{qubit}_ef": device_setup.logical_signal_groups[qubit].logical_signals[
        "drive_line_ef"
    ],
    f"drive_{qubit}": device_setup.logical_signal_groups[qubit].logical_signals["drive_line"],
    "measure": device_setup.logical_signal_groups[qubit].logical_signals[
        "measure_line"
    ],
    "acquire": device_setup.logical_signal_groups[qubit].logical_signals[
        "acquire_line"
    ],
}

spin_locking_exp = create_spin_locking_exp()
spin_locking_exp.set_signal_map(signal_map_default)

exp_calibration = Calibration()

exp_calibration[f"drive_{qubit}_ef"] = SignalCalibration(
    oscillator=Oscillator(
        frequency=freq_sweep, modulation_type=ModulationType.HARDWARE
    ),
)
spin_locking_exp.set_calibration(exp_calibration)

# %% compile
session = Session(device_setup=device_setup)
session.connect(do_emulation=False)
compiler_settings = {"SHFSG_MIN_PLAYWAVE_HINT": 128, "SHFSG_MIN_PLAYZERO_HINT": 128}
compiled_exp = session.compile(spin_locking_exp, compiler_settings=compiler_settings)

if simulate:
    plot_simulation(compiled_exp, 0e-6, 3e-6, signals=[f"drive_{qubit}_ef", "measure"])

# %% run
qspec_results = session.run()

# %% acquire data
acquired_results_X = qspec_results.get_data("handle_X")
# acquired_results_Y = qspec_results.get_data("handle_Y")
# acquired_results_Z = qspec_results.get_data("handle_Z")


# %% 1d plot
ts = delay_sweep.values


guess = [1, 1e-6, 0, 1e-6, 0.5]

amplitude_X = np.abs(acquired_results_X[0])

params, params_covariance = curve_fit(cos_wave_exp, ts, amplitude_X, p0=guess)

# freq = 1 / params[1] * 1e-6
# T2 = params[3]

# print(f"freq = {freq} MHz")
# print(f"T2 = {T2 * 1e6} us")

plt.plot(delay_sweep.values * 1e6, amplitude_X)
plt.plot(delay_sweep.values * 1e6, cos_wave_exp(delay_sweep.values, *params))
plt.title('Spin Locking 1d Plot')
plt.ylim([-0.2, 1.2])
plt.xlabel("Time [us]")
plt.ylabel("Amplitude [a.u.]")
plt.show()


# %% 2d plot


x,y = np.meshgrid(delay_sweep.values, detunings)

plt.title('Spin Locking 2d Plot')

z = np.abs(acquired_results_X)
print(z.shape)
plt.pcolormesh(x*1e6, y*1e-6, z)
plt.colorbar()

plt.show()



# %% save

meta_data = {
    "type": "2d",
    "plot_properties": None,
    "experiment_properties": {
        "qubit": qubit,
        "repetitions": exp_repetitions,
        "ramp_pulse": amp,
        "ramp_length": ramp_length,
        "f_rabi": f_rabi,
        "detunings": list(detunings),
        "qubit_parameters": qubit_parameters[qubit],
    },
}

# data = {
#     "t": delay_sweep.values,
#     "f": freq_sweep.values,
#     "x_data_r": acquired_results_X.real,
#     "x_data_c": acquired_results_X.imag,
#     "y_data_r": acquired_results_Y.real,
#     "y_data_c": acquired_results_Y.imag,
#     "z_data_r": acquired_results_Z.real,
#     "z_data_c": acquired_results_Z.imag,
# }

exp_data = ExperimentData(meta_data=meta_data, data=data)
# exp_data.save_data(file_name="spin_locking")

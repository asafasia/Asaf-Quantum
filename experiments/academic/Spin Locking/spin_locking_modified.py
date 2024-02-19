import numpy as np
# import qutip
from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from helper import pulses, exp_helper
from laboneq.simple import *
import spin_locking_utils as ut
from qubit_parameters import qubit_parameters
from helper.experiment_results import ExperimentData
from matplotlib import pyplot as plt
from helper.utility_functions import correct_axis, cos_wave_exp
from helper.kernels import kernels
from scipy.fft import fft, fftfreq
import labber.labber_util as lu
from helper.exp_helper import *
from helper.pulses import *
import logging

# %% class

def find_new_phase(qubit, detunings, f_rabi):
    phases = []
    for detuning in detunings:
        phases.append(
            ut.calculate_expected_freq(qubit, detuning, f_rabi))

    return phases


def furrier_transform(ts, x):
    N = len(ts)
    dt = ts[1] - ts[0]

    yf = 2 / N * abs(fft(x)[0:N // 2])
    xf = fftfreq(len(ts), dt)[:N // 2]
    max_freq = xf[np.argmax(np.abs(yf[1:])) + 1]
    return xf, yf, max_freq


class SpinLockingExperiment:
    def __init__(
            self,
            qubit,
            mode,
            delay_start,
            delay_stop,
            delay_step,
            phase,
            base_phase,
            measure_basis,
            exp_repetitions=1000,
            use_correction_matrix=False,
    ):
        self.results_2d_R = None
        self.results_1d_X = None
        self.results_1d_Y = None
        self.results_1d_Z = None

        self.qubit_parameters = qubit_parameters[qubit]
        self.max_freq = None
        self.results_2d_Z = None
        self.results_2d_Y = None
        self.results_2d_X = None
        self.detunings = None
        self.results_2d = None
        self.results = None
        self.measure_basis = measure_basis
        self.acquired_results_Y = None
        self.acquired_results_Z = None
        self.acquired_results_X = None
        self.detuning = None
        self.experiment = None
        self.use_correction_matrix = use_correction_matrix
        self.qubit = qubit
        self.phase = phase
        self.base_phase = base_phase
        self.mode = mode
        modulation_type = 'hardware' if mode == 'spec' else 'software'
        if mode == 'spec':
            acquisition_type = AcquisitionType.SPECTROSCOPY
        elif mode == 'int':
            acquisition_type = AcquisitionType.INTEGRATION
        elif mode == 'disc':
            acquisition_type = AcquisitionType.DISCRIMINATION
        else:
            raise ValueError("mode must be one of 'spec', 'int', 'disc'")
        self.acquisition_type = acquisition_type

        self.kernel = pulses.readout_pulse(qubit) if mode == 'spec' else kernels[qubit]

        initializer = exp_helper.initialize_exp()
        self.device_setup = initializer.create_device_setup(modulation_type=modulation_type)

        self.exp_signals = initializer.signals(qubit)
        self.signal_map_default = initializer.signal_map_default(qubit)

        self.ramp_length = 200e-9
        self.qubit_relax = 200e-6
        self.simulate = False
        self.exp_repetitions = exp_repetitions

        self.f_rabi = 90e6

        self.amp = 2 * qubit_parameters[qubit]["pi_amp"] * qubit_parameters[qubit]["pi_len"] * self.f_rabi

        self.delay_start = delay_start
        self.delay_step = delay_step
        self.delay_stop = delay_stop

        self.delay_sweep = LinearSweepParameter(start=delay_start, stop=self.delay_stop, count=self.delay_step)

    def new_delay_sweep(self, start, stop, step):
        self.delay_start = start
        self.delay_step = step
        self.delay_stop = stop

        self.delay_sweep = LinearSweepParameter(start=start, stop=stop, count=step)

    def spin_lock(self, detuning, phase, exp, qubit: str, basis: str = "Z", ):
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
                pulse=ut.ramp_pulse(qubit, amplitude=self.amp),
                pulse_parameters={"detuning": detuning, "ramp_state": "up"},
                length=self.ramp_length,
            )
        #
        with exp.section(uid=f"delay_{basis}", play_after=f"up_{basis}"):
            exp.play(
                signal=f"drive_{qubit}_ef",
                pulse=ut.delay_pulse(qubit, amplitude=self.amp),
                length=self.delay_sweep,
            )
        #
        with exp.section(uid=f"down_{basis}"):
            exp.play(
                signal=f"drive_{qubit}_ef",
                pulse=ut.ramp_pulse(qubit, amplitude=self.amp),
                pulse_parameters={"detuning": detuning, "ramp_state": "down"},
                length=self.ramp_length,
            )
            exp.delay(signal=f"drive_{qubit}_ef", time=10e-9)

        with exp.section(uid=f"virtual_z_gate_{basis}", play_after=f"down_{basis}"):
            exp.play(
                signal=f"drive_{qubit}",
                pulse=None,
                increment_oscillator_phase=-2 * np.pi * (phase - self.base_phase) * self.delay_sweep

            )

        with exp.section(uid=f"change_basis_{basis}", play_after=f"virtual_z_gate_{basis}"):
            if basis == "Z":
                exp.play(
                    signal=f"drive_{qubit}",
                    pulse=pulses.pi_pulse(qubit),
                    amplitude=0,
                    phase=0,
                )
            elif basis == "X":
                exp.play(
                    signal=f"drive_{qubit}",
                    pulse=pulses.pi_pulse(qubit),
                    amplitude=1 / 2,
                    phase= -np.pi / 2 #- 2 * np.pi * (phase - self.base_phase) * self.delay_sweep,
                )
            elif basis == "Y":
                exp.play(
                    signal=f"drive_{qubit}",
                    pulse=pulses.pi_pulse(qubit),
                    amplitude=1 / 2,
                    phase=np.pi #- 2 * np.pi * (phase - self.base_phase) * self.delay_sweep,
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
                kernel=self.kernel,
            )

        with exp.section(
                uid=f"relax_{basis}",
        ):
            exp.delay(
                signal="measure",
                time=self.qubit_relax,
            )

    def define_experiment(self, detuning, phase=None):
        if not phase:
            phase = self.phase
        self.detuning = detuning
        spin_locking_exp = Experiment(
            uid="Spin_locking_experiment",
            signals=[
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
                ExperimentSignal(f"drive_{self.qubit}_ef"),
                ExperimentSignal(f"drive_{self.qubit}"),
            ],
        )

        with spin_locking_exp.acquire_loop_rt(
                uid="RT_shots",
                count=self.exp_repetitions,
                acquisition_type=self.acquisition_type,
                reset_oscillator_phase=True,

        ):
            with spin_locking_exp.sweep(uid="rabi_sweep", parameter=self.delay_sweep):
                for basis in self.measure_basis:
                    with spin_locking_exp.section(uid=f"basis_{basis}"):
                        self.spin_lock(detuning, phase, spin_locking_exp, self.qubit, basis=basis)

        signal_map_default = {
            f"drive_{self.qubit}_ef": self.device_setup.logical_signal_groups[self.qubit].logical_signals[
                "drive_line_ef"
            ],
            f"drive_{self.qubit}": self.device_setup.logical_signal_groups[self.qubit].logical_signals["drive_line"],
            "measure": self.device_setup.logical_signal_groups[self.qubit].logical_signals[
                "measure_line"
            ],
            "acquire": self.device_setup.logical_signal_groups[self.qubit].logical_signals[
                "acquire_line"
            ],
        }

        spin_locking_exp.set_signal_map(signal_map_default)

        exp_calibration = Calibration()

        exp_calibration[f"drive_{self.qubit}_ef"] = SignalCalibration(
            oscillator=Oscillator(
                frequency=detuning + qubit_parameters[self.qubit]['qb_freq'] - qubit_parameters[self.qubit]['qb_lo'],
                modulation_type=ModulationType.HARDWARE
            ),
        )
        spin_locking_exp.set_calibration(exp_calibration)

        self.experiment = spin_locking_exp

    def run(self):
        session = Session(device_setup=self.device_setup, log_level=logging.WARNING)
        session.connect(do_emulation=False)
        compiler_settings = {"SHFSG_MIN_PLAYWAVE_HINT": 128, "SHFSG_MIN_PLAYZERO_HINT": 128}
        compiled_exp = session.compile(self.experiment, compiler_settings=compiler_settings)

        if self.simulate:
            plot_simulation(compiled_exp, 0e-6, 3e-6, signals=[f"drive_{self.qubit}_ef", "measure"])

        self.results = session.run()

        for basis in self.measure_basis:
            if basis == 'X':
                self.results_1d_X = self.results.get_data(f"handle_X")
            elif basis == 'Y':
                self.results_1d_Y = self.results.get_data(f"handle_Y")
            elif basis == 'Z':
                self.results_1d_Z = self.results.get_data(f"handle_Z")

    def run_2d_experiment(self, detunings, phases):
        self.results_2d_X = []
        self.results_2d_Y = []
        self.results_2d_Z = []

        self.detunings = detunings
        for i, detuning in enumerate(detunings):

            print(f"Running detuning {i + 1}/{len(detunings)} \n detuning = {detuning * 1e-6:.1f} MHz")
            print(f"base phase = {self.base_phase * 1e-6:.1f} MHz \n current phase : {phases[i]*1e-6:.1f} MHz" )

            self.define_experiment(detuning=detuning, phase=phases[i])
            self.run()
            for basis in self.measure_basis:
                res = list(abs(self.results.get_data(f"handle_{basis}")))
                if basis == 'X':
                    self.results_2d_X.append(res)
                elif basis == 'Y':
                    self.results_2d_Y.append(res)
                elif basis == 'Z':
                    self.results_2d_Z.append(res)

            X_temp = np.array(self.results_2d_X)
            Y_temp = np.array(self.results_2d_Y)
            Z_temp = np.array(self.results_2d_Z)

            if len(self.measure_basis) == 3:
                self.results_2d_R = np.sqrt(
                    (2 * X_temp - 1) ** 2 + (2 * Y_temp - 1) ** 2 + (
                            2 * Z_temp - 1) ** 2)

    def plot_2d(self, basis='X'):
        if basis == 'X':
            z = self.results_2d_X
        elif basis == 'Y':
            z = self.results_2d_Y
        elif basis == 'Z':
            z = self.results_2d_Z
        else:
            raise ValueError("basis must be one of 'X', 'Y', 'Z'")
        x, y = np.meshgrid(spin_locking_exp.delay_sweep.values, self.detunings)

        print(np.shape(x))

        plt.pcolormesh(x * 1e6, y * 1e-6, z)
        plt.title(f'Spin Locking 2d Plot : {basis} basis')
        plt.xlabel("Time [us]")
        plt.ylabel("Detuning [MHz]")
        plt.colorbar()
        plt.show()

    def furrier_transform_1d(self, basis='X', plot=True):
        if len(self.measure_basis) == 1:
            basis = self.measure_basis[0]

        acquired_results = self.results.get_data(f"handle_{basis}")
        ts = self.delay_sweep.values
        amplitude = np.abs(acquired_results)

        xf, yf, self.max_freq = furrier_transform(ts, amplitude)

        if plot:
            plt.title(f'Fourier Transform : {basis} basis')
            plt.xlabel("Frequency [MHz]")
            plt.ylabel("Amplitude [a.u.]")
            plt.plot(xf[1:] * 1e-6, yf[1:],
                     label=f'data (detuning = {self.detuning * 1e-6:.1f} MHz)')
            plt.axvline(x=self.max_freq * 1e-6, color='r', linestyle='--',
                        label=f'freq = {self.max_freq * 1e-6:.1f} MHz')
            plt.legend()

            plt.show()
        return xf[1:], yf[1:]

    def furrier_transform_2d(self, basis='X'):
        ts = self.delay_sweep.values
        z = []
        if basis == 'X':
            res = self.results_2d_X
        elif basis == 'Y':
            res = self.results_2d_Y
        elif basis == 'Z':
            res = self.results_2d_Z
        else:
            raise ValueError("basis must be one of 'X', 'Y', 'Z'")

        max_freq_vec = []

        for i, detuning in enumerate(self.detunings):
            ws, a, max_freq = furrier_transform(ts, res[i])
            z.append(a)
            max_freq_vec.append(max_freq)

        max_freq_vec = np.array(max_freq_vec)

        # predicted_freqs = find_new_phase(self.qubit, self.detunings, self.f_rabi)
        x, y = np.meshgrid(ws, self.detunings)
        z = np.array(z)

        plt.plot(max_freq_vec * 1e-6, self.detunings * 1e-6, 'o', color='red')

        # plt.plot(np.array(predicted_freqs) * 1e-6, detunings * 1e-6, 'o', color='orange')


        plt.title('Fourier Transform 2d Plot')
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Detuning [MHz]")
        plt.pcolormesh(x[:, 1:] * 1e-6, y[:, 1:] * 1e-6, z[:, 1:])
        plt.colorbar()
        plt.show()

        return max_freq_vec, self.detunings

    def plot_1d(self, basis='X', fit=False):

        if len(self.measure_basis) == 1:
            basis = self.measure_basis[0]

        self.acquired_results = self.results.get_data(f"handle_{basis}")

        ts = self.delay_sweep.values
        amplitude = np.abs(self.acquired_results)

        plt.plot(ts * 1e6, amplitude, label=f'data (detuning = {self.detuning * 1e-6:.1f} MHz)')
        plt.plot(ts * 1e6, amplitude, 'o', markersize=3, color='green')

        if fit:
            guess = [1, 1e-6, 0, 1e-6, 0.5]
            params = curve_fit(cos_wave_exp, ts, amplitude, p0=guess)[0]

            freq = 1 / params[1] * 1e-6
            T2 = params[3]

            print(f"freq = {freq} MHz")
            print(f"T2 = {T2 * 1e6} us")
            x = np.linspace(ts[0], ts[-1], 1000)
            plt.plot(x * 1e6, cos_wave_exp(x, *params), label=f'fit (T2 = {params[3] * 1e6:.3f} us)', alpha=0.7)

        plt.title(f'Spin Locking 1d Plot: {basis} basis')
        plt.ylim([0, 1])
        plt.xlabel("Time [us]")
        plt.ylabel("Amplitude [a.u.]")
        plt.legend()
        plt.show()

    def plot_bloch_sphere(self):
        if len(self.measure_basis) == 3:
            acquired_results_X = self.qspec_results.get_data(f"handle_X")
            acquired_results_Y = self.qspec_results.get_data(f"handle_Y")
            acquired_results_Z = self.qspec_results.get_data(f"handle_Z")

            amplitude_X = np.abs(acquired_results_X)
            amplitude_Y = np.abs(acquired_results_Y)
            amplitude_Z = np.abs(acquired_results_Z)

            b = qutip.Bloch()
            pnt = [2 * amplitude_X - 1, 2 * amplitude_Y - 1, 2 * amplitude_Z - 1]
            b.add_points(pnt)
            b.render()
            plt.show()
        else:
            print("Can only plot bloch sphere for all 3 basis")

    def save_to_labber_1d(self):
        measured_data = {}
        for basis in self.measure_basis:
            if basis == 'X':
                measured_data[basis] = self.results_1d_X
            elif basis == 'Y':
                measured_data[basis] = self.results_1d_Y
            elif basis == 'Z':
                measured_data[basis] = self.results_1d_Z
        # measured_data = {'=self.results_1d_X, Y=self.results_1d_Y, Z=self.results_1d_Z)

        measured_data = dict(X=self.results_1d_X)
        sweep_parameters = dict(delay=np.array(self.delay_sweep.values))
        units = dict(delay="s")

        meta_data = dict(detuning=self.detuning, tags=["Nadav-Lab", "spin-locking"], user="Guy", f_rabi=self.f_rabi,
                         phase=self.phase,
                         base_phase=self.base_phase, qubit=self.qubit, qubit_parameters=self.qubit_parameters,
                         pulse_library=f"{ut.date}/run{ut.run_index}")

        exp_result = dict(measured_data=measured_data, sweep_parameters=sweep_parameters, units=units,
                          meta_data=meta_data)

        lu.create_logfile("spin_locking", **exp_result, loop_type="1d")

    def save_to_labber_2d(self):
        measured_data = {}
        for basis in self.measure_basis:
            if basis == 'X':
                measured_data[basis] = np.array(self.results_2d_X)
            elif basis == 'Y':
                measured_data[basis] = np.array(self.results_2d_Y)
            elif basis == 'Z':
                measured_data[basis] = np.array(self.results_2d_Z)

        if len(self.measure_basis) == 3:
            measured_data['R'] = np.array(self.results_2d_R)

        sweep_parameters = dict(delay=np.array(self.delay_sweep.values), detuning=np.array(self.detunings))
        units = dict(delay="s", detuning="Hz")

        meta_data = dict(tags=["Nadav-Lab", "spin-locking"], user="Guy", f_rabi=self.f_rabi, phase=self.phase,
                         base_phase=self.base_phase, qubit=self.qubit, qubit_parameters=self.qubit_parameters,
                         pulse_library=f"{ut.date}/run{ut.run_index}")

        exp_result = dict(measured_data=measured_data, sweep_parameters=sweep_parameters, units=units,
                          meta_data=meta_data)

        lu.create_logfile("spin_locking", **exp_result, loop_type="2d")


if __name__ == '__main__':

    args = {
        'qubit': 'q5',
        'mode': 'disc',
        'delay_start': 10e-9,
        'delay_stop': 1e-6,
        'delay_step': 251,  # max around 500
        'exp_repetitions': 100,
        'base_phase': 1e6,
        'phase':0,

        'measure_basis': ['X','Y','Z'],
        'use_correction_matrix': False,

    }

    single_run = False

    detunings = ut.get_detunings()

    spin_locking_exp = SpinLockingExperiment(**args)
    if single_run:
        detuning_index = 45
        phase = ut.calculate_expected_freq(spin_locking_exp.qubit, detunings[detuning_index], f_rabi=spin_locking_exp.f_rabi)
        # phase = 0.0
        spin_locking_exp.define_experiment(detuning=detunings[detuning_index], phase=phase)

        spin_locking_exp.run()

        spin_locking_exp.plot_1d(basis='X', fit=False)
        spin_locking_exp.furrier_transform_1d(basis='X')
        # spin_locking_exp.plot_bloch_sphere()
        spin_locking_exp.save_to_labber_1d()



    else:
        detunings = detunings[0:-1:60]

        # detunings_for_interp = detunings
        # #
        phases = find_new_phase(spin_locking_exp.qubit, detunings, spin_locking_exp.f_rabi)
        # phases = detunings*0
        # spin_locking_exp.run_2d_experiment(detunings_for_interp, phases)
        # spin_locking_exp.plot_2d(basis='X')
        # v = spin_locking_exp.furrier_transform_2d(basis='X')
        # v = np.array(v)
        # np.savetxt('interp_data.txt', v)
        #
        # # %% interp
        # v = np.loadtxt('interp_data.txt')
        # interpolation_function = interp1d(v[1], v[0], kind='linear')
        #
        # # detunings = detunings[0:-1:20]
        # phases = find_new_phase(spin_locking_exp.qubit, detunings, spin_locking_exp.f_rabi)
        #
        # new_phases = interpolation_function(detunings)

        # %% again with good phases(freqs)
        # spin_locking_exp.new_delay_sweep(start=0, stop=1e-6, step=401)

        spin_locking_exp.run_2d_experiment(detunings, phases)
        spin_locking_exp.plot_2d(basis='X')
        spin_locking_exp.furrier_transform_2d(basis='X')
        spin_locking_exp.save_to_labber_2d()
        # %%
        #
        # def fourier_plot(experiment, basis='X'):
        #     ts = experiment.delay_sweep.values
        #     z = []
        #     if basis == 'X':
        #         res = experiment.results_2d_X
        #     elif basis == 'Y':
        #         res = experiment.results_2d_Y
        #     elif basis == 'Z':
        #         res = experiment.results_2d_Z
        #     else:
        #         raise ValueError("basis must be one of 'X', 'Y', 'Z'")
        #
        #     max_freq_vec = []
        #
        #     for i, detuning in enumerate(experiment.detunings):
        #         ws, a, max_freq = furrier_transform(ts, res[i])
        #         z.append(a)
        #         max_freq_vec.append(max_freq)
        #
        #     max_freq_vec = np.array(max_freq_vec)
        #
        #     predicted_freqs = find_new_phase(experiment.qubit, experiment.detunings, experiment.f_rabi)
        #     x, y = np.meshgrid(ws, experiment.detunings)
        #     z = np.array(z)
        #
        #     plt.plot(max_freq_vec * 1e-6, experiment.detunings * 1e-6, 'o', color='red')
        #
        #     plt.plot(np.array(predicted_freqs) * 1e-6, detunings * 1e-6, 'o', color='orange')
        #
        #     plt.title('Fourier Transform 2d Plot')
        #     plt.xlabel("Frequency [MHz]")
        #     plt.ylabel("Detuning [MHz]")
        #     plt.pcolormesh(x[:, 1:] * 1e-6, y[:, 1:] * 1e-6, z[:, 1:])
        #     plt.colorbar()
        #     plt.xlim([28, 70])
        #     plt.legend(["measured", "predicted"])
        #     plt.show()
        #
        #     return max_freq_vec, experiment.detunings
        #
        # fourier_plot(spin_locking_exp,basis='X')



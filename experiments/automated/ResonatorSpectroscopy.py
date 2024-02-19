import numpy as np
import matplotlib.pyplot as plt
from laboneq.simple import *
from qubit_parameters import qubit_parameters, update_qp
from experiments.calibrations.resonator_spectroscopy import resonator_spectroscopy
from helper.experiment import QuantumExperiment1Q


# from play_flux.qdac_test import play_flux_qdac


class ResonatorSpectroscopy(QuantumExperiment1Q):
    def __init__(self, q, span, amp, exp_repetitions=1000, simulate=False, do_emulation=True):
        super().__init__(q, simulate, do_emulation)
        self.span = span
        self.amp = amp
        self.exp_repetitions = exp_repetitions
        self._add_experiment()

    def _add_experiment(self):
        q = self.qubit

        central_frequency = qubit_parameters[q]["res_freq"]

        dfs = np.linspace(
            start=central_frequency - self.span / 2,
            stop=central_frequency + self.span / 2,
            num=200
        )

        res_LO = qubit_parameters[q]["res_lo"]
        freq_sweep = SweepParameter(uid='freq_RF_sweep', values=dfs - res_LO)

        exp = resonator_spectroscopy(
            qubit=q,
            signals=self.exp_signals,
            exp_repetitions=self.exp_repetitions,
            freq_sweep=freq_sweep,
            long_pulse=True
        )

        exp_calibration = Calibration()

        exp_calibration["measure"] = SignalCalibration(
            oscillator=Oscillator(
                "readout_osc",
                frequency=freq_sweep,
                modulation_type=ModulationType.HARDWARE,
            ),
        )

        exp.set_calibration(exp_calibration)

        exp.set_signal_map(self.signal_map_default)

        self.experiment = exp

    def plot(self):
        acquire_results_1 = self.results.get_data("spec_1")
        acquire_results_2 = self.results.get_data("spec_2")

        dfs = self.results.get_axis("spec_1")[0] + qubit_parameters[qubit]["res_lo"]

        amplitude_1 = np.abs(acquire_results_1)
        amplitude_2 = np.abs(acquire_results_2)

        phase_radians_1 = np.unwrap(np.angle(acquire_results_1)) * amplitude_1
        phase_radians_2 = np.unwrap(np.angle(acquire_results_2)) * amplitude_2

        # Convert res_freq to GHz

        # Create a figure with 2 subplots stacked vertically
        fig, axes = plt.subplots(3, 1, figsize=(8, 6))
        drive_amp = qubit_parameters[qubit]['drive_amp']
        res_amp = qubit_parameters[qubit]['res_amp']
        res_len = qubit_parameters[qubit]['res_len']

        fig.suptitle(
            f'Resonator Spectroscopy {qubit} \n drive amp = {drive_amp} V \n res amp = {res_amp:.2f} V \n res len = {res_len * 1e6} us',
            fontsize=18)

        # Plot the amplitude in the first subplot
        axes[0].plot(dfs * 1e-9, amplitude_1, color='blue', marker='.', label='with_drive')
        axes[0].plot(dfs * 1e-9, amplitude_2, color='green', marker='.', label='without_drive')

        axes[0].set_xlabel('Frequency [GHz]')
        axes[0].set_ylabel('Amplitude [a.u.]')
        axes[0].grid(True)

        # Plot the phase in radians in the second subplot
        axes[1].plot(dfs * 1e-9, phase_radians_1, color='red', marker='.', label='with_drive')
        axes[1].plot(dfs * 1e-9, phase_radians_2, color='yellow', marker='.', label='without_drive')
        axes[1].set_xlabel('Frequency [GHz]')
        axes[1].set_ylabel('Phase [rad]')
        axes[1].grid(True)
        # axes[0].axvline(x=qubit_parameters[qubit]["res_freq"] * 1e-9, color='green', linestyle='--')
        # axes[1].axvline(x=qubit_parameters[qubit]["res_freq"] * 1e-9, color='green', linestyle='--')

        axes[1].legend()
        axes[0].legend()

        axes[2].plot(dfs * 1e-9, np.abs(amplitude_1 - amplitude_2), color='blue', marker='.', label='with_drive')
        # axes[2].axvline(x=qubit_parameters[qubit]["res_freq"] * 1e-9, color='green', linestyle='--')

        plt.tight_layout()
        plt.show()

    def update(self):

        acquire_results_1 = self.results.get_data("spec_1")
        acquire_results_2 = self.results.get_data("spec_2")

        dfs = self.results.get_axis("spec_1")[0]

        amplitude_1 = np.abs(acquire_results_1)
        amplitude_2 = np.abs(acquire_results_2)

        diff = np.abs(amplitude_1 - amplitude_2)
        df_var = np.sqrt(np.var(diff))
        max_index = np.argmax(diff)

        max_f = dfs[max_index]

        print(f"res freq = {dfs[max_index] * 1e-9} GHz")
        if max(diff) / df_var > 10:
            user_input = input('press y to update n to not update:')
            print('updated !!!!!!!!!!')
            if user_input.lower() == "y":
                update_qp(qubit, 'res_freq', max_f)
                print("Variable updated!")
            elif user_input.lower() == "n":
                print("No updates were made.")
            else:
                raise ValueError("Invalid input.")
        else:
            print('not updated !!!!!!!')


if __name__ == "__main__":
    qubit = "q4"

    rs = ResonatorSpectroscopy(q=qubit, span=200e6, amp=0.5, do_emulation=False, exp_repetitions=100)

    rs.run()

    rs.plot()

    rs.update()

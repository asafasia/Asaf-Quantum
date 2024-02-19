import numpy as np
import matplotlib.pyplot as plt
from laboneq.simple import *
from scipy.optimize import curve_fit
from helper.utility_functions import correct_axis, cos_wave
from qubit_parameters import qubit_parameters, update_qp
from helper.experiment import QuantumExperiment1Q
from experiments.calibrations.amplitude_rabi import power_rabi_experiment


class PowerRabi(QuantumExperiment1Q):

    def __init__(self, q: str, pis_number=3, exp_repetitions=1000, steps=101, simulate=False, do_emulation=True):
        super().__init__(q, simulate, do_emulation)
        self.pis_number = pis_number
        self.exp_repetitions = exp_repetitions
        self.steps = steps
        self._add_experiment()

    def _add_experiment(self):
        amp_sweep = LinearSweepParameter(start=0, stop=1, count=self.steps)
        exp = power_rabi_experiment(
            qubit=self.qubit,
            amplitude_sweep=amp_sweep,
            exp_signals=self.exp_signals,
            exp_repetitions=self.exp_repetitions,
            pis_number=self.pis_number
        )
        exp.set_signal_map(self.signal_map_default)
        self.experiment = exp

    def plot(self):
        amplitude = correct_axis(np.abs(self.results.get_data("amp_rabi")), qubit_parameters[self.qubit]["ge"])
        rabi_amp = self.results.get_axis("amp_rabi")[0] * qubit_parameters[self.qubit]["pi_amp"] * self.pis_number

        guess = [(max(amplitude) - min(amplitude)) / 2, 0.2, np.pi / 2, np.mean(amplitude)]

        self.fit_params = curve_fit(cos_wave, rabi_amp, amplitude, p0=guess)[0]

        new_amp = abs(self.fit_params[1]) / 2

        print("new amplitude = ", new_amp)

        plt.plot(rabi_amp, amplitude, '-', color='black', label='data')
        plt.plot(rabi_amp, cos_wave(rabi_amp, *self.fit_params),
                 label=f'Fit: pi amplitude = {self.fit_params[1]}', color='red')
        plt.xlabel('Drive Amplitude [a.u.]')
        plt.ylabel('Amplitude [a.u.]')
        plt.legend()
        plt.title(f'Amplitude Rabi {qubit}', fontsize=18)
        plt.tight_layout()
        plt.show()

    def update(self):
        new_amp = self.fit_params[1]
        if not abs(self.do_emulation and new_amp - qubit_parameters[qubit]['pi_amp']) < 0.3:
            user_input = input("Do you want to update the pi amplitude? [y/n]")
            if user_input == 'y':
                update_qp(qubit, 'pi_amp', new_amp)
            elif user_input == 'n':
                print("No update")
            else:
                raise Exception("Invalid input")
        else:
            print("No update")


if __name__ == "__main__":
    qubit = "q5"

    power_rabi = PowerRabi(q=qubit, pis_number=3, steps=101, do_emulation=False, exp_repetitions=1000)

    power_rabi.run()

    power_rabi.plot()

    power_rabi.update()

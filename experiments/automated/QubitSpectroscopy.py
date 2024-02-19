import numpy as np
import matplotlib.pyplot as plt
from laboneq.simple import *
from helper import pulses
from helper.utility_functions import correct_axis
from qubit_parameters import qubit_parameters, update_qp
from helper.experiment import QuantumExperiment1Q


def inv_parabola(frequency, qubit):
    a, b, c = qubit_parameters[qubit]['parabola']

    f1 = -b + np.sqrt(b ** 2 - 4 * a * c - 4 * a * (frequency - c)) / (2 * a)
    f2 = -b - np.sqrt(b ** 2 - 4 * a * c - 4 * a * (frequency - c)) / (2 * a)

    return min(abs(f1), abs(f2))


class QubitSpectroscopy(QuantumExperiment1Q):

    def __init__(self, q: str, span, amp, exp_repetitions=1000, simulate=False, do_emulation=True):
        super().__init__(q, simulate, do_emulation)
        self.span = span
        self.amp = amp
        self.exp_repetitions = exp_repetitions
        self._add_experiment()

    def _add_experiment(self, steps=101):
        q = self.qubit

        drive_LO = qubit_parameters[q]["qb_lo"]
        center = qubit_parameters[q]["qb_freq"]

        IF_dfs = np.linspace(start=center - self.span / 2, stop=center + self.span / 2,
                             num=steps)  # for carrier calculation
        freq_sweep = SweepParameter(uid=f'freq_sweep_{q}', values=IF_dfs - drive_LO)  # sweep object

        exp_qspec = Experiment(
            uid="Qubit Spectroscopy",
            signals=self.exp_signals,
        )
        with exp_qspec.acquire_loop_rt(
                uid="freq_shots",
                count=self.exp_repetitions,
                acquisition_type=AcquisitionType.SPECTROSCOPY,
        ):
            with exp_qspec.sweep(uid="freq_sweep", parameter=freq_sweep):
                with exp_qspec.section(uid="qubit_excitation"):
                    exp_qspec.play(signal=f"drive_{q}", pulse=pulses.spec_pulse(q), amplitude=self.amp)

                with exp_qspec.section(
                        uid="readout_section", play_after="qubit_excitation", trigger={"measure": {"state": True}}
                ):
                    exp_qspec.play(signal="measure",
                                   pulse=pulses.readout_pulse(q),
                                   phase=qubit_parameters[q]['angle'])
                    exp_qspec.acquire(
                        signal="acquire",
                        handle="qubit_spec",
                        length=qubit_parameters[q]["res_len"],
                        # kernel = kernel
                    )
                with exp_qspec.section(uid="delay"):
                    exp_qspec.delay(signal="measure", time=120e-6)

        exp_calibration = Calibration()

        exp_calibration[f"drive_{q}"] = SignalCalibration(
            oscillator=Oscillator(
                frequency=freq_sweep,
                modulation_type=ModulationType.HARDWARE,
            ),
        )

        exp_qspec.set_calibration(exp_calibration)
        exp_qspec.set_signal_map(self.signal_map_default)
        self.experiment = exp_qspec

    def update(self, update_type):
        qubit = self.qubit
        center = qubit_parameters[qubit]["qb_freq"]
        results = self.results.get_data("qubit_spec")
        amplitude = np.real(results)
        amplitude = correct_axis(amplitude, qubit_parameters[qubit]["ge"])
        dfs = self.results.get_axis("qubit_spec")[0] + qubit_parameters[qubit]["qb_lo"]

        max_frequency = dfs[np.argmax(amplitude)]

        print(f"max frequency = {max_frequency * 1e-9} GHz")
        if not self.do_emulation:
            if max(amplitude) / np.var(amplitude) > 10:
                print("max frequency > noise")
                if update_type == "frequency":
                    update_qp(qubit, 'qb_freq', max_frequency)
                elif update_type == "flux":
                    flux_bias_update = inv_parabola(max_frequency, qubit)
                    update_qp(qubit, 'flux_bias', flux_bias_update)
                else:
                    raise ValueError("update type is not frequency or flux")

            else:
                print("max frequency < noise")
        else:
            print("update could not happen because emulation mode")

    def plot(self):
        qubit = self.qubit
        center = qubit_parameters[qubit]["qb_freq"]
        results = self.results.get_data("qubit_spec")
        amplitude = np.abs(results)
        # amplitude = correct_axis(amplitude, qubit_parameters[qubit]["ge"])
        dfs = self.results.get_axis("qubit_spec")[0] + qubit_parameters[qubit]["qb_lo"]

        drive_amp = qubit_parameters[qubit]['drive_amp'] * self.amp

        flux_bias = qubit_parameters[qubit]['flux_bias']

        new_min = dfs[np.argmax(amplitude)]

        print(f"new min freq = {new_min * 1e-9} GHz")
        w0 = True

        if w0:
            plt.title(
                f'Qubit Spectroscopy {qubit} w0\n drive amp = {drive_amp:5f} V \n flux bias = {flux_bias:.5f} V \nresonanance = {center * 1e-6} MHz  ',
                fontsize=18)
        else:
            plt.title(
                f'Qubit Spectroscopy {qubit} w1\n drive amp = {drive_amp:5f} V \n flux bias = {flux_bias:.5f} V \nresonanance = {center * 1e-6:.2f} MHz  ',
                fontsize=18)
        plt.axvline(x=0, color='green', linestyle='--')
        # plt.ylim([-0.1, 1.1])
        plt.plot(dfs * 1e-6 - center * 1e-6, amplitude, "k")
        # plt.plot(dfs*1e-6 - center*1e-6,func_lorentz(dfs,*popt))
        plt.xlabel('Detuning [MHz]')
        plt.ylabel('Amplitude [a.u.]')
        plt.show()


if __name__ == "__main__":
    qubit = "q5"
    qs = QubitSpectroscopy(q=qubit, span=2e6, amp=1/400, exp_repetitions=500, simulate=False, do_emulation=False)

    qs.run()

    qs.plot()

    # qs.update()

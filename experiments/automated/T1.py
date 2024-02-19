import numpy as np
import matplotlib.pyplot as plt
from laboneq.simple import *
from helper import pulses
from helper.utility_functions import correct_axis
from qubit_parameters import qubit_parameters, update_qp
from helper.experiment import QuantumExperiment1Q

class QubitSpectroscopy(QuantumExperiment1Q):

    def __init__(self, q: str, sweep_start, sweep_start, step_num, exp_repetitions=1000, simulate=False):
        super().__init__(q, simulate)
        self.span = span
        self.amp = amp
        self.exp_repetitions = exp_repetitions
        self._add_experiment()

    def _add_experiment(self, steps=101):
        q = self.qubit

        ts = np.linspace(0, sweep_stop, step_num)
        delay_sweep = SweepParameter(values=ts)

        t1_exp = Experiment(
            uid="Amplitude Rabi",
            signals=exp_signals
        )
        with t1_exp.acquire_loop_rt(
                uid="rabi_shots",
                count=exp_repetitions,
                acquisition_type=AcquisitionType.SPECTROSCOPY,
        ):
            # inner loop - real time sweep of Rabi ampitudes
            with t1_exp.sweep(uid="rabi_sweep", parameter=delay_sweep):
                # play qubit excitation pulse - pulse amplitude is swept
                with t1_exp.section(
                        uid="qubit_excitation"
                ):
                    t1_exp.play(
                        signal=f"drive_{qubit}", pulse=pulses.pi_pulse(qubit),
                    )
                    t1_exp.delay(signal=f"measure", time=delay_sweep)
                # readout pulse and data acquisition
                with t1_exp.section(uid="readout_section", play_after="qubit_excitation"):
                    # play readout pulse on measure line
                    t1_exp.play(signal="measure",
                                pulse=pulses.readout_pulse(qubit),
                                phase=qubit_parameters[qubit]['angle'])
                    # trigger signal data acquisition
                    t1_exp.acquire(
                        signal="acquire",
                        handle="t1_handle",
                        length=qubit_parameters[qubit]["res_len"],
                    )
                with t1_exp.section(uid="delay"):
                    # relax time after readout - for qubit relaxation to groundstate and signal processing
                    t1_exp.delay(signal="measure", time=120e-6)


        exp_calibration = Calibration()
        t1_exp.set_calibration(exp_calibration)
        t1_exp.set_signal_map(self.signal_map_default)
        self.experiment = t1_exp

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
        amplitude = np.real(results)
        amplitude = correct_axis(amplitude, qubit_parameters[qubit]["ge"])
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
    qubit = "q1"
    qs = QubitSpectroscopy(q=qubit, span=100e6, amp=0.5)

    qs.run()

    qs.plot()

    qs.update()

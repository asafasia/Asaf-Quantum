import cmath
import numpy as np
import matplotlib.pyplot as plt
from laboneq.simple import *
from helper import pulses
from qubit_parameters import qubit_parameters, update_qp
from helper.experiment import QuantumExperiment1Q


class IQDiscrimination(QuantumExperiment1Q):
    def __init__(self, q: str, exp_repetitions=10000, simulate=False,do_emulation=True):
        super().__init__(q, simulate,do_emulation)
        self.exp_repetitions = exp_repetitions
        self._add_experiment()

    def _add_experiment(self):
        exp = Experiment(
            uid="Optimal weights",
            signals=self.exp_signals
        )

        with exp.acquire_loop_rt(
                uid="shots",
                count=self.exp_repetitions,  # pow(2, average_exponent),
                averaging_mode=AveragingMode.SINGLE_SHOT,
                acquisition_type=AcquisitionType.SPECTROSCOPY,

        ):
            with exp.section(uid='ground'):
                with exp.section():
                    exp.play(signal=f"drive_{self.qubit}", pulse=pulses.pi_pulse(self.qubit), amplitude=0)

                with exp.section():
                    with exp.section():
                        exp.play(signal="measure",
                                 pulse=pulses.readout_pulse(self.qubit),
                                 phase=qubit_parameters[self.qubit]['angle'])
                        exp.acquire(
                            signal="acquire", handle="ac_g", length=pulses.readout_pulse(self.qubit).length,
                        )

                    with exp.section():
                        exp.delay(signal="measure", time=120e-6)
            with exp.section(uid='excited'):
                with exp.section():
                    exp.play(signal=f"drive_{self.qubit}", pulse=pulses.pi_pulse(self.qubit), amplitude=1)

                with exp.section():
                    with exp.section():
                        exp.play(signal="measure",
                                 pulse=pulses.readout_pulse(self.qubit),
                                 phase=qubit_parameters[self.qubit]['angle'])
                        exp.acquire(
                            signal="acquire", handle="ac_e", length=pulses.readout_pulse(self.qubit).length,
                        )

                    with exp.section():
                        exp.delay(signal="measure", time=120e-6)

        exp.set_signal_map(self.signal_map_default)
        self.experiment = exp

    def plot(self, exp_type='clouds'):
        raw_0 = np.array(self.results.get_data("ac_g"))
        raw_1 = np.array(self.results.get_data("ac_e"))

        traces = np.array([raw_0, raw_1])

        m1 = np.mean(traces[0])
        m2 = np.mean(traces[1])
        v1 = np.var(traces[0])
        v2 = np.var(traces[1])

        self.angle = -cmath.phase(m2 - m1)
        separation = abs((m2 - m1) / np.sqrt(v1))

        self.m1 = m1
        self.m2 = m2

        if exp_type == 'clouds':
            plt.plot(traces[0].real, traces[0].imag, '.', alpha=0.6, label='ground')
            plt.plot(traces[1].real, traces[1].imag, '.', alpha=0.6, label='excited')

            # mean
            plt.plot(m1.real, m1.imag, 'x', color='black')
            plt.plot(m2.real, m2.imag, 'x', color='black')

            plt.title(f'IQ Clouds {qubit}')
            plt.xlabel('I [a.u.]', )
            plt.ylabel('Q [a.u.]')
            plt.legend()

            plt.show()

        elif exp_type == 'histogram':
            res_freq = qubit_parameters[qubit]['res_freq']
            res_amp = qubit_parameters[qubit]['res_amp']
            res_len = qubit_parameters[qubit]['res_len']

            plt.title(
                f"Seperation Histogram {qubit} \nSeperation = {separation:.3f} \nres_freq = {res_freq * 1e-6:.2f} MHz  \nres amp = {res_amp} V \nres len = {res_len * 1e6} us ")
            plt.hist(traces[0].real, bins=80, edgecolor='black', label='ground state', alpha=0.6)
            plt.hist(traces[1].real, bins=80, edgecolor='black', label='excited state', alpha=0.5)

            plt.xlabel('I [a.u.]')
            plt.ylabel('number')
            plt.legend()
            plt.axvline(x=m2 / 2 + m1 / 2, linestyle='--', color='black')
            plt.axvline(x=m1, linestyle='--', color='blue')
            plt.axvline(x=m2, linestyle='--', color='red')

            plt.show()

            print(f'separation = {separation:.3f}')
        else:
            raise ValueError(f"Unknown experiment type: {exp_type}")

    def update(self):
        if not self.do_emulation:
            update_qp(qubit, 'angle', self.angle + qubit_parameters[qubit]['angle'])
            update_qp(qubit, 'ge', [self.m1.real, self.m2.real])


if __name__ == "__main__":
    qubit = "q4"

    iq_discrimination = IQDiscrimination(qubit, 20000, simulate=True,do_emulation=False)
    iq_discrimination.run()
    iq_discrimination.plot(exp_type='clouds')
    iq_discrimination.plot(exp_type='histogram')
    iq_discrimination.update()

from experiments.automated.QubitSpectroscopy import QubitSpectroscopy
from experiments.automated.ResonatorSpectroscopy import ResonatorSpectroscopy
from experiments.automated.PowerRabi import PowerRabi
from experiments.automated.IQDiscrimination import IQDiscrimination

if __name__ == '__main__':
    qubit = "q1"
    # %% resonator spectroscopy

    rs = ResonatorSpectroscopy(q=qubit, span=100e6, amp=0.5)
    rs.run()
    rs.update()

    # %% qubit spectroscopy

    span = 100e6
    amp = 0.5

    for i in range(5):
        qs = QubitSpectroscopy(q=qubit, span=span, amp=amp)

        qs.run()

        qs.update(update_type='flux')

        span = span / 2
        amp = amp / 2

    # %% power rabi

    pr = PowerRabi(q=qubit)
    pr.run()
    pr.update()

    # %% iq discrimination

    iqd = IQDiscrimination(q=qubit)
    iqd.run()
    iqd.update()

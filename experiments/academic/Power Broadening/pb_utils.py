import numpy as np
from laboneq.dsl.experiment import pulse_library

from qubit_parameters import qubit_parameters

CAN_COMPRESS = True


@pulse_library.register_pulse_functional
def lorentzian(x, p, n, **_):
    a = np.sqrt((1 / p) ** (1 / n) - 1)
    return 1 / (1 + (a * x) ** 2) ** n


@pulse_library.register_pulse_functional
def lorentzian_half(x, p, n, **_):
    a = np.sqrt((1 / p) ** (1 / n) - 1)
    f = 1 / (1 + (a * x) ** 2) ** n

    return f - 2 * np.heaviside(x, 1) * f


def power_broadening_pulse(
        qubit,
        amplitude=None,
        length=None,
        pulse_type="Square",
        p=1 / 10,
        n=2 / 3,
        half_pulse=False
):
    if not amplitude:
        amplitude = qubit_parameters[qubit]["pi_amp"]

    if not length:
        length = qubit_parameters[qubit]["pi_len"]

    if pulse_type == "Square":
        pulse = pulse_library.const(
            uid=f"pi_pulse_{qubit}", length=length, amplitude=amplitude
        )

    elif pulse_type == "Gaussian":
        pulse = pulse_library.gaussian(
            uid=f"pi_pulse_{qubit}",
            sigma=np.sqrt(-np.log(p) / 2),
            length=length,
            amplitude=amplitude,
        )

    elif pulse_type == "Lorentzian":
        if not half_pulse:
            pulse = lorentzian(
                uid=f"pi_pulse_{qubit}",
                length=length,
                amplitude=amplitude,
                p=p,
                n=n,
                can_compress=CAN_COMPRESS
            )
        else:
            pulse = lorentzian_half(
                uid=f"pi_pulse_{qubit}",
                length=length,
                amplitude=amplitude,
                p=p,
                n=n,
                can_compress=CAN_COMPRESS

            )
    else:
        raise ValueError("pulse_type must be Square, Gaussian or Lorentzian")

    return pulse

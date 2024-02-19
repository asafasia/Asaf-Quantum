from laboneq.simple import *
import numpy as np
from qubit_parameters import qubit_parameters

CAN_COMPRESS = True


@pulse_library.register_pulse_functional
def lorentizan(x, p, n, **_):
    a = np.sqrt((1 / p) ** (1 / n) - 1)
    return 1 / (1 + (a * x) ** 2) ** n


@pulse_library.register_pulse_functional
def k_pulse(x, **_):
    return np.heaviside(x - 0.5, 0.5) * 0


def pi_pulse(qubit):
    return pulse_library.const(
        uid=f"pi_pulse_{qubit}",
        length=qubit_parameters[qubit]["pi_len"],
        amplitude=qubit_parameters[qubit]["pi_amp"],
        can_compress=CAN_COMPRESS,
    )


def many_pi_pulse(qubit, pis: int):
    return pulse_library.const(
        uid=f"pi_pulse_{qubit}",
        length=qubit_parameters[qubit]["pi_len"],
        amplitude=qubit_parameters[qubit]["pi_amp"] * pis,
    )


def power_broadening_pulse(
        qubit,
        amplitude=None,
        length=None,
        pulse_type="Square",
        p=1 / 10,
        n=2 / 3,
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
        pulse = lorentizan(
            uid=f"pi_pulse_{qubit}", length=length, amplitude=amplitude, p=p, n=n
        )
    else:
        raise ValueError("pulse_type must be Square, Gaussian or Lorentzian")

    return pulse


def kernel_pulse(qubit):
    return pulse_library.const(
        uid=f"kernel_pulse_{qubit}",
        length=qubit_parameters[qubit]["res_len"],
        amplitude=qubit_parameters[qubit]["res_amp"],
    )


def spec_pulse(qubit):
    return pulse_library.const(
        uid=f"spec_pulse_{qubit}",
        length=qubit_parameters[qubit]["drive_len"],
        amplitude=qubit_parameters[qubit]["drive_amp"],
        can_compress=CAN_COMPRESS,
    )


def readout_pulse(qubit):
    return pulse_library.const(
        uid=f"readout_pulse_{qubit}",
        length=qubit_parameters[qubit]["res_len"],
        amplitude=qubit_parameters[qubit]["res_amp"],
    )


def flux_pulse(qubit, amplitude=None):
    if amplitude:
        amp = amplitude
    else:
        amp = qubit_parameters[qubit]["flux_bias"]

    return pulse_library.const(
        uid=f"flux_pulse_{qubit}",
        length=120e-6,
        amplitude=amp,

    )

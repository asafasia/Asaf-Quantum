from laboneq.simple import *
import numpy as np
from qubit_parameters import qubit_parameters

CAN_COMPRESS = True


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

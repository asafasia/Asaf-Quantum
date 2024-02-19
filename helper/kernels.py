import numpy as np
from numpy.typing import NDArray
from zhinst.utils.shfqa.multistate import QuditSettings
from laboneq.dsl.experiment import pulse_library as pl


def calculate_integration_kernels(
        state_traces: list[NDArray],
) -> list[pl.PulseSampledComplex]:
    """Calculates the optimal kernel arrays for state discrimination given a set of
    reference traces corresponding to the states. The calculated kernels can directly be
    used as kernels in acquire statements.

    Args:
        state_traces: List of complex-valued reference traces, one array per state. The
            reference traces are typically obtained by an averaged scope measurement of
            the readout resonator response when the qudit is prepared in a certain
            state.

    """

    n_traces = len(state_traces)
    settings = QuditSettings(state_traces)

    weights = settings.weights[: n_traces - 1]
    return [pl.sampled_pulse_complex(weight.vector) for weight in weights]

from .project_path import project_path


traces_q1 = np.loadtxt(f"{project_path}/helper/kernels/traces_q1.txt", dtype=complex)
traces_q2 = np.loadtxt(f"{project_path}/helper/kernels/traces_q2.txt", dtype=complex)
traces_q3 = np.loadtxt(f"{project_path}/helper/kernels/traces_q3.txt", dtype=complex)
traces_q4 = np.loadtxt(f"{project_path}/helper/kernels/traces_q4.txt", dtype=complex)
traces_q5 = np.loadtxt(f"{project_path}/helper/kernels/traces_q5.txt", dtype=complex)

kernels = {"q1": calculate_integration_kernels(traces_q1),
           "q2": calculate_integration_kernels(traces_q2),
           "q3": calculate_integration_kernels(traces_q3),
           "q4": calculate_integration_kernels(traces_q4),
           "q5": calculate_integration_kernels(traces_q5)
           }


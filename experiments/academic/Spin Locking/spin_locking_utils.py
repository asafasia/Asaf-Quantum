import numpy as np
from laboneq.contrib.example_helpers.plotting.plot_helpers import *
from laboneq.simple import *
from scipy.interpolate import interp1d
from helper import pulses, exp_helper
from qubit_parameters import qubit_parameters
# from qutip import *

import pandas as pd

qubit = "q5"
run_index = 1
qb_freq = qubit_parameters[qubit]["qb_freq"]

date = "2024_01_25"


def get_detunings():
    path = f"pulse_library/{date}/run_{run_index}"
    file = "params.csv"
    path = f"{path}/{file}"
    df = pd.read_csv(path)

    samples = df["detuning_scan"].to_numpy()
    return samples * 1e6


def get_pulse(ramp: str, i: int):
    path = f"pulse_library/{date}/run_{run_index}"
    ramp_up = "control_ramp_up"
    ramp_down = "control_ramp_down"
    if ramp == "up":
        path = f"{path}/{ramp_up}_{i}.csv"
    else:
        path = f"{path}/{ramp_down}_{i}.csv"

    samples_up = pd.read_csv(path, usecols=[1, 3, 4]).to_numpy()
    t = samples_up[:, 0]
    x = samples_up[:, 1]
    y = samples_up[:, 2]
    return t, x / max(abs(x)), y / max(abs(x))


def interp_pulse(t, x, y):
    t = t / 100 - 1
    func_x = interp1d(t, x, kind="linear")
    func_y = interp1d(t, y, kind="linear")

    return func_x, func_y


def delay_pulse(qubit: str, amplitude=1, length=1e-6):
    delay_pulse = pulse_library.const(
        uid=f"pulse_{qubit}", length=length, amplitude=amplitude, can_compress=True
    )

    return delay_pulse


def ramp_pulse(qubit: str, amplitude=1, length=1e-6, simple=False):
    if simple:
        ramp_pulse = pulse_library.const(
            uid=f"ramp_{qubit}", length=length, amplitude=amplitude, can_compress=True
        )
    else:
        ds = get_detunings()
        enumerated_dict = {value: index for index, value in enumerate(ds)}

        @pulse_library.register_pulse_functional
        def ramp(x, detuning: float, ramp_state: str, **_):
            i = enumerated_dict[detuning]
            t_vec, x_vec, y_vec = get_pulse(ramp_state, i)
            func_x, func_y = interp_pulse(t_vec, x_vec, y_vec)
            return func_x(x) + 1j * func_y(x)

        ramp_pulse = ramp(
            uid=f"ramp_pulse_{qubit}",
            length=length,
            amplitude=amplitude,
        )
    return ramp_pulse


def calculate_expected_freq(qubit, detuning, f_rabi):
    N = 10
    alpha = qubit_parameters[qubit]["anharmonicity"]
    a = annihil_op(N)

    H = -detuning * a.T @ a - 0.5 * alpha * a.T @ a.T @ a @ a + 0.5 * f_rabi * (a + a.T)

    E = np.linalg.eig(H)[0]
    E_min = min(abs(np.diff(E)))

    return E_min-detuning


def annihil_op(N):
    vec = np.sqrt(np.arange(1, N ))
    a= np.diag(vec, 1)
    return a
print(calculate_expected_freq('q5', 50e6, 70e6))
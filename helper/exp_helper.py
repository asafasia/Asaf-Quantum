from laboneq.simple import *
import numpy as np
from pathlib import Path
import time
from numpy.typing import NDArray
from laboneq.dsl.experiment import pulse_library as pl
from zhinst.utils.shfqa.multistate import QuditSettings
from .qdac import play_flux_qdac

from pprint import pprint
from laboneq.contrib.example_helpers.generate_example_datastore import generate_device_setup
# import Labber

from qubit_parameters import qubit_parameters

from .descriptor import descriptor


def define_calibration(device_setup, parameters, modulation_type="hardware"):
    if modulation_type == "hardware":
        mt = ModulationType.HARDWARE
    elif modulation_type == "software":
        mt = ModulationType.SOFTWARE
    else:
        print("enter correct modulation type")

    # Define LOs
    def single_lo_oscillator(uid, qubit, lo_type):

        if qubit[0] != 'c':
            oscillator = Oscillator()
            oscillator.uid = f"{uid}" + f"{qubit}" + "_osc"
            oscillator.frequency = qubit_parameters[qubit][lo_type]

            return oscillator

    readout_lo_dict = {
        k: single_lo_oscillator("readout_lo_", k, "res_lo")
        for k in device_setup.logical_signal_groups.keys()
    }

    drive_lo_dict = {
        k: single_lo_oscillator("drive_lo_", k, "qb_lo")
        for k in device_setup.logical_signal_groups.keys()
    }

    calibration = Calibration()

    for qubit in device_setup.logical_signal_groups.keys():

        if qubit[0] != "c":

            qb_freq = qubit_parameters[qubit]["qb_freq"]
            qb_lo = qubit_parameters[qubit]["qb_lo"]
            qb_if = -qb_lo + qb_freq

            ro_freq = qubit_parameters[qubit]["res_freq"]
            ro_lo = qubit_parameters[qubit]["res_lo"]
            ro_if = -ro_lo + ro_freq

            calibration[
                device_setup.logical_signal_groups[qubit].logical_signals["acquire_line"]
            ] = SignalCalibration(
                # oscillator=Oscillator(
                #     uid = f'acquire_if_{qubit}',
                #     frequency= ro_if,
                #     modulation_type=ModulationType.SOFTWARE,
                # ),
                local_oscillator=Oscillator(
                    uid=f"acquire_lo_{qubit}",
                    frequency=ro_lo,

                ),
                threshold=qubit_parameters[qubit]['threshold'],
                port_delay=80e-9
            )

            ro_calib = SignalCalibration(
                oscillator=Oscillator(
                    uid=f"readout_if_{qubit}",
                    frequency=ro_if,
                    modulation_type=mt,
                ),

                local_oscillator=Oscillator(
                    uid=f"ro_lo_{qubit}",
                    frequency=ro_lo,
                    modulation_type=mt
                ),
                range=5

            )

            drive_calib = SignalCalibration(
                oscillator=Oscillator(
                    uid=f"drive_if_{qubit}",
                    frequency=qb_if,
                    modulation_type=ModulationType.HARDWARE
                ),
                local_oscillator=drive_lo_dict[qubit],
                range=10

            )

            drive_calib_ef = SignalCalibration(
                oscillator=Oscillator(
                    uid=f"drive_if_{qubit}_ef",
                    frequency=qb_if,
                    modulation_type=ModulationType.HARDWARE
                ),
                local_oscillator=drive_lo_dict[qubit],
                range=10

            )

            calibration[
                device_setup.logical_signal_groups[qubit].logical_signals[
                    "measure_line"
                ]

            ] = ro_calib

            calibration[
                device_setup.logical_signal_groups[qubit].logical_signals[
                    "drive_line"
                ]
            ] = drive_calib

            calibration[
                device_setup.logical_signal_groups[qubit].logical_signals[
                    "drive_line_ef"
                ]
            ] = drive_calib_ef



        else:
            flux_calib = SignalCalibration(
                voltage_offset=qubit_parameters[qubit]["flux_bias"]  # {V}
            )
            calibration[
                device_setup.logical_signal_groups[qubit].logical_signals[
                    "flux_line"
                ]
            ] = flux_calib

    return calibration


class initialize_exp:
    def __init__(self):
        self.device_setup = None
        self.calibrations = None

    def create_device_setup(self, modulation_type='hardware'):
        self.device_setup = DeviceSetup.from_descriptor(
            yaml_text=descriptor,
            server_host="127.0.0.1",
            server_port="8004",  # port number of the dataserver - default is 8004
            setup_name="my_setup",  # setup name
        )

        self._add_default_calibrations(modulation_type)
        self._play_flux_bias()

        return self.device_setup

    def _play_flux_bias(self):
        # for i, qubit in enumerate(self.device_setup.logical_signal_groups.keys()):
        #     play_flux_qdac(qubit=i+1, flux_bias=qubit_parameters[qubit]['flux_bias'])

        play_flux_qdac(qubit='q1', flux_bias=qubit_parameters['q1']['flux_bias'])
        play_flux_qdac(qubit='q2', flux_bias=qubit_parameters['q2']['flux_bias'])
        play_flux_qdac(qubit='q3', flux_bias=qubit_parameters['q3']['flux_bias'])
        play_flux_qdac(qubit='q4', flux_bias=qubit_parameters['q4']['flux_bias'])
        play_flux_qdac(qubit='q5', flux_bias=qubit_parameters['q5']['flux_bias'])
        pass

    def _add_default_calibrations(self, modulation_type='hardware'):
        calibrations = define_calibration(self.device_setup,
                                          qubit_parameters,
                                          modulation_type
                                          )
        self.device_setup.set_calibration(calibrations)
        self.calibrations = calibrations

    def signal_map_default(self, qubit=None):

        if not qubit:
            signal_map = {
                "measure_q1": self.device_setup.logical_signal_groups["q1"].logical_signals["measure_line"],
                "measure_q2": self.device_setup.logical_signal_groups["q2"].logical_signals["measure_line"],
                "measure_q3": self.device_setup.logical_signal_groups["q3"].logical_signals["measure_line"],
                "measure_q4": self.device_setup.logical_signal_groups["q4"].logical_signals["measure_line"],
                "measure_q5": self.device_setup.logical_signal_groups["q5"].logical_signals["measure_line"],

                "acquire_q1": self.device_setup.logical_signal_groups["q1"].logical_signals["acquire_line"],
                "acquire_q2": self.device_setup.logical_signal_groups["q2"].logical_signals["acquire_line"],
                "acquire_q3": self.device_setup.logical_signal_groups["q3"].logical_signals["acquire_line"],
                "acquire_q4": self.device_setup.logical_signal_groups["q4"].logical_signals["acquire_line"],
                "acquire_q5": self.device_setup.logical_signal_groups["q5"].logical_signals["acquire_line"],

                "flux_c13": self.device_setup.logical_signal_groups["c13"].logical_signals["flux_line"],
                "flux_c23": self.device_setup.logical_signal_groups["c23"].logical_signals["flux_line"],
                "flux_c43": self.device_setup.logical_signal_groups["c43"].logical_signals["flux_line"],
                "flux_c53": self.device_setup.logical_signal_groups["c53"].logical_signals["flux_line"],

                "drive_q1": self.device_setup.logical_signal_groups["q1"].logical_signals["drive_line"],
                "drive_q2": self.device_setup.logical_signal_groups["q2"].logical_signals["drive_line"],
                "drive_q3": self.device_setup.logical_signal_groups["q3"].logical_signals["drive_line"],
                "drive_q4": self.device_setup.logical_signal_groups["q4"].logical_signals["drive_line"],
                "drive_q5": self.device_setup.logical_signal_groups["q5"].logical_signals["drive_line"],

            }
        else:
            signal_map = {
                "measure": self.device_setup.logical_signal_groups[qubit].logical_signals["measure_line"],
                "acquire": self.device_setup.logical_signal_groups[qubit].logical_signals["acquire_line"],

                "flux_c13": self.device_setup.logical_signal_groups["c13"].logical_signals["flux_line"],
                "flux_c23": self.device_setup.logical_signal_groups["c23"].logical_signals["flux_line"],
                "flux_c43": self.device_setup.logical_signal_groups["c43"].logical_signals["flux_line"],
                "flux_c53": self.device_setup.logical_signal_groups["c53"].logical_signals["flux_line"],

                "drive_q1": self.device_setup.logical_signal_groups["q1"].logical_signals["drive_line"],
                "drive_q2": self.device_setup.logical_signal_groups["q2"].logical_signals["drive_line"],
                "drive_q3": self.device_setup.logical_signal_groups["q3"].logical_signals["drive_line"],
                "drive_q4": self.device_setup.logical_signal_groups["q4"].logical_signals["drive_line"],
                "drive_q5": self.device_setup.logical_signal_groups["q5"].logical_signals["drive_line"],

            }

        return signal_map

    def signals(self, qubit=None):
        if not qubit:
            signals = [

                ExperimentSignal("drive_q1"),
                ExperimentSignal("drive_q2"),
                ExperimentSignal("drive_q3"),
                ExperimentSignal("drive_q4"),
                ExperimentSignal("drive_q5"),

                ExperimentSignal("flux_c13"),
                ExperimentSignal("flux_c23"),
                ExperimentSignal("flux_c43"),
                ExperimentSignal("flux_c53"),

                ExperimentSignal("measure_q1"),
                ExperimentSignal("measure_q2"),
                ExperimentSignal("measure_q3"),
                ExperimentSignal("measure_q4"),
                ExperimentSignal("measure_q5"),

                ExperimentSignal("acquire_q1"),
                ExperimentSignal("acquire_q2"),
                ExperimentSignal("acquire_q3"),
                ExperimentSignal("acquire_q4"),
                ExperimentSignal("acquire_q5"),
            ]

        else:
            signals = [

                ExperimentSignal("drive_q1"),
                ExperimentSignal("drive_q2"),
                ExperimentSignal("drive_q3"),
                ExperimentSignal("drive_q4"),
                ExperimentSignal("drive_q5"),

                ExperimentSignal("flux_c13"),
                ExperimentSignal("flux_c23"),
                ExperimentSignal("flux_c43"),
                ExperimentSignal("flux_c53"),

                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ]

        return signals


# drive_amp = qubit_parameters["q3"]["drive_amp"]
# drive_len = qubit_parameters["q3"]["drive_len"]
# res_amp= qubit_parameters["q3"]["res_amp"]
# measure_length = qubit_parameters["q3"]["res_len"]


# flux_pulse_length = 120e-6#drive_len+2*measure_length
# amplitude_flux = 1

# flux_pulse = pulse_library.const(
#     uid="flux_spec_pulse_c13", length=flux_pulse_length, amplitude=amplitude_flux
# )


def create_drive_freq_sweep(qubit, start_freq, stop_freq, num_points):
    return LinearSweepParameter(
        uid=f"drive_freq_{qubit}", start=start_freq, stop=stop_freq, count=num_points
    )


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


def save_func(session, file_name):
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    Path("Results").mkdir(parents=True, exist_ok=True)
    session.save_results(f"Results/{timestamp}_coupler_spec.json")


def correct_axis(amplitudes, ge):
    return (amplitudes - ge[0]) / (ge[1] - ge[0])


# %%load kernel for qubits

# for qubit_key, qubit_info in qubit_parameters.items():
#     # Check if the current entry is for a qubit
#     if qubit_info.get("qb_freq") is not None:
#         # Generate the file name based on the qubit key
#         traces_filename = f"traces_{qubit_key}"

#     if traces_filename is not None:
#         # Load the kernel from the file

#         kernel = np.load(traces_filename)
#         globals()[f"kernel_{qubit_key}"] = kernel

#         print(f"Kernel for {qubit_key} loaded as {kernel_name}")
import numpy as np



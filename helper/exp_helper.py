from laboneq.simple import *
from .qdac import play_flux_qdac
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



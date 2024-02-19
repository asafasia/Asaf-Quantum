from laboneq.contrib.example_helpers.plotting.plot_helpers import *
from laboneq._utils import id_generator
from qiskit import qasm3, QuantumCircuit
from math import pi
from helper.exp_helper import *
from helper.kernels import kernels
from helper.pulses import *
from qubit_parameters import qubit_parameters


def drive_pulse(qubit: Qubit, label):
    return pulse_library.const(
        uid=f"{qubit.uid}_{label}",
        length=qubit.parameters.user_defined["pulse_length"],
        amplitude=qubit.parameters.user_defined["amplitude_pi"],
    )


def drive_pulse_root(qubit: Qubit, label):
    return pulse_library.const(
        uid=f"{qubit.uid}_{label}",
        length=qubit.parameters.user_defined["pulse_length"],
        amplitude=qubit.parameters.user_defined["amplitude_pi"] / 2
    )


def rz(qubit: Qubit):
    def rz_gate(angle: float):
        gate = Section(uid=id_generator(f"p_{qubit.uid}_rz_{int(180 * angle / pi)}"))
        gate.play(
            signal=qubit.signals["drive"],
            pulse=None,
            increment_oscillator_phase=angle,
        )
        return gate

    return rz_gate


def measurement(qubit: Qubit, kernel):
    def measurement_gate(handle: str):
        gate = Section(uid=id_generator(f"meas_{qubit.uid}_{handle}"),
                       trigger={"measure": {"state": True}})

        gate.reserve(signal=qubit.signals["drive"])

        gate.play(signal=qubit.signals["measure"], pulse=readout_pulse(qubit.uid),
                  phase=qubit.parameters.user_defined["angle"])

        gate.acquire(
            signal=qubit.signals["acquire"],
            handle=handle,
            kernel=kernel,
        )
        gate.delay(signal=qubit.signals["measure"], time=120e-6)

        return gate

    return measurement_gate


# def cz(control: Qubit, target: Qubit):
#     """Return a controlled X gate for the specified control and target qubits.
#
#     The CX gate function takes no arguments and returns a LabOne Q section that performs
#     the controllex X gate.
#     """
#
#     def cz_gate():
#         cz_id = f"cz_{control.uid}_{target.uid}"
#
#         qubit_pair = (control.uid, target.uid)
#         if qubit_pair in coupler_map:
#             coupler = coupler_map[qubit_pair]
#             c_object = coupler_dict[coupler]
#             # Assuming a function to get the coupler object
#         else:
#             raise ValueError(f"No coupler found for qubits {control.uid} and {target.uid}")
#
#         gate = Section(uid=id_generator(cz_id))
#
#         flux_section = Section(uid=id_generator('flux_pulse'))
#
#         flux_section.play(
#             signal=c_object.signals["flux"], pulse=flux_pulse(c_object.uid), length=80e-9
#         )
#         # print(c_object.signals)
#
#         gate.add(flux_section)
#
#         phase_shift_cancel_section = Section(uid=id_generator(f"p_{control.uid}_phase_shift_cancellation"),
#                                              play_after=flux_section)
#
#         phase_shift_cancel_section.play(
#             signal=control.signals["drive"],
#             pulse=None,
#             increment_oscillator_phase=qubit_parameters[control.uid]['cz_phase_shift'],
#         )
#
#         phase_shift_cancel_section.play(
#             signal=target.signals["drive"],
#             pulse=None,
#             increment_oscillator_phase=qubit_parameters[target.uid]['cz_phase_shift'],
#         )
#         gate.add(phase_shift_cancel_section)
#
#         return gate
#
#     return cz_gate


class QuantumProcessor:
    def __init__(self, mode,qubits):
        self.gubit_map = {}
        self.results = None
        self.compiled_experiment = None
        self.mode = mode
        self.modulation_type = 'hardware' if mode == 'spec' else 'software'
        if mode == 'spec':
            self.acquisition_type = AcquisitionType.SPECTROSCOPY
        elif mode == 'int':
            self.acquisition_type = AcquisitionType.INTEGRATION
        elif mode == 'disc':
            self.acquisition_type = AcquisitionType.DISCRIMINATION
        self.qubits = qubits
        self.kernels = {qubit: readout_pulse(qubit) if mode == 'spec' else kernels[qubit] for qubit in self.qubits}
        print(self.kernels)
        exp = initialize_exp()
        self.device_setup = exp.create_device_setup(self.modulation_type)
        self.session = Session(device_setup=self.device_setup)
        self.session.connect(do_emulation=False)
        self.gate_store = GateStore()
        self.qubit_map = None
        self.add_qubits()

        self.add_gates()

    def add_qubits(self):
        qubit_map = {}
        for i, qubit in enumerate(self.qubits):
            qubit_obj = Transmon.from_logical_signal_group(
                qubit,
                lsg=self.device_setup.logical_signal_groups[qubit],
                parameters=TransmonParameters(
                    user_defined={
                        "amplitude_pi": qubit_parameters[qubit]['pi_amp'],
                        "pulse_length": qubit_parameters[qubit]['pi_len'],
                        "readout_len": qubit_parameters[qubit]['res_len'],
                        "readout_amp": qubit_parameters[qubit]['res_amp'],
                        "reset_delay_length": 100e-6,
                        "reset_length": 0,
                        "angle": qubit_parameters[qubit]['angle'],
                    },
                ),
            )

            qubit_map[f'q[{i}]'] = qubit_obj
        self.qubit_map = qubit_map

    def add_gates(self):

        for oq3_qubit, l1q_qubit in self.qubit_map.items():
            self.gate_store.register_gate(
                "sx",
                oq3_qubit,
                drive_pulse_root(l1q_qubit, label="sx"),
                signal=l1q_qubit.signals["drive"],
            )
            self.gate_store.register_gate(
                "x",
                oq3_qubit,
                drive_pulse(l1q_qubit, label="x"),
                signal=l1q_qubit.signals["drive"],
            )
            self.gate_store.register_gate_section("rz", (oq3_qubit,), rz(l1q_qubit))
            self.gate_store.register_gate_section("measure", (oq3_qubit,),
                                                  measurement(l1q_qubit, self.kernels[l1q_qubit.uid]))

    def add_experiment(self, qasm_circs):
        exp = exp_from_qasm_list(
            qasm_circs,
            qubits=self.qubit_map,
            gate_store=self.gate_store,
            repetition_time=200e-6,
            batch_execution_mode="rt",
            do_reset=False,
            count=1000,
            pipeline_chunk_count=1,
            acquisition_type=self.acquisition_type,
        )

        self.compiled_experiment = self.session.compile(exp)

    def run_experiment(self):
        self.results = self.session.run(self.compiled_experiment)
        return self.results

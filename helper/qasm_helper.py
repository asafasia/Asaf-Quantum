

from helper.exp_helper import *
from helper.kernels import kernels
from helper.pulses import *
from qubit_parameters import qubit_parameters

coupler_map = {
    ("q1", "q3"): "c13",
    ("q3", "q1"): "c13",
    ("q2", "q3"): "c23",
    ("q3", "q2"): "c23",
    ("q4", "q3"): "c43",
    ("q3", "q4"): "c43",
    ("q5", "q3"): "c53",
    ("q3", "q5"): "c53",
}
couplers = ["c13", "c23", "c43", "c53"]





class QuantumProcessor:
    def __init__(self, mode, qubits, pipeline_chunk_count=2):
        self.coupler_map = None
        self.gubit_map = {}
        self.pipeline_chunk_count = pipeline_chunk_count
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
        self.couplers = ["c13", "c23", "c43", "c53"]
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

        coupler_map = {}
        for i, coupler in enumerate(couplers):
            c_object = Transmon.from_logical_signal_group(
                uid=coupler,
                lsg=self.device_setup.logical_signal_groups[coupler],
                parameters=TransmonParameters(

                    user_defined={
                        "coupler_flux": qubit_parameters[coupler]['flux_bias'],
                    },
                ),
            )
            coupler_map[f'c[{i}]'] = c_object
        self.coupler_map = coupler_map

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

        if len(self.qubits) > 0:
            self.gate_store.register_gate_section("cz",
                                                  ("q[0]", "q[1]"),
                                                  cz(self.qubit_map["q[0]"], self.qubit_map["q[1]"],
                                                     self.coupler_map["c[0]"]))
            print(self.coupler_map)
            self.qubit_map.update(self.coupler_map)

    def add_experiment(self, qasm_circs):
        exp = exp_from_qasm_list(
            qasm_circs,
            qubits=self.qubit_map,
            gate_store=self.gate_store,
            repetition_time=200e-6,
            batch_execution_mode="pipeline",
            do_reset=False,
            count=1000,
            pipeline_chunk_count=self.pipeline_chunk_count,
            acquisition_type=self.acquisition_type,
        )

        self.compiled_experiment = self.session.compile(exp)

    def run_experiment(self):
        self.results = self.session.run(self.compiled_experiment)
        return self.results

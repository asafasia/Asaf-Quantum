from laboneq.contrib.example_helpers.plotting.plot_helpers import *
from laboneq._utils import id_generator
from qiskit import qasm3, QuantumCircuit
from math import pi
from helper.kernels import kernels
from helper.exp_helper import *
from helper.pulses import *
from qubit_parameters import qubit_parameters

qubit = 'q5'

mode = 'int'
modulation_type = 'hardware' if mode == 'spec' else 'software'
if mode == 'spec':
    acquisition_type = AcquisitionType.SPECTROSCOPY
elif mode == 'int':
    acquisition_type = AcquisitionType.INTEGRATION
elif mode == 'disc':
    acquisition_type = AcquisitionType.DISCRIMINATION

kernel = readout_pulse(qubit) if mode == 'spec' else kernels[qubit]

exp = initialize_exp()

device_setup = exp.create_device_setup(modulation_type)

exp_signals = exp.signals(qubit)
signal_map_default = exp.signal_map_default(qubit)

q = Transmon.from_logical_signal_group(
    qubit,
    lsg=device_setup.logical_signal_groups[qubit],
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

gate_store = GateStore()

qubit_map = {"q[0]": q}


# %%
def drive_pulse(qubit: Qubit, label, length=50e-9, amplitude=0.6):
    """Return a drive pulse for the given qubit.

    In practice different drive pulses would be specified for each qubit and operation.
    """
    return pulse_library.const(
        uid=f"{qubit.uid}_{label}",
        length=qubit.parameters.user_defined["pulse_length"],
        amplitude=qubit.parameters.user_defined["amplitude_pi"],
    )


def drive_pulse_root(qubit: Qubit, label, length=50e-9, amplitude=0.6):
    """Return a root drive pulse for the given qubit.

    In practice different drive pulses would be specified for each qubit and operation.
    """
    return pulse_library.const(
        uid=f"{qubit.uid}_{label}",
        length=qubit.parameters.user_defined["pulse_length"],
        amplitude=qubit.parameters.user_defined["amplitude_pi"] / 2

    )


def rz(qubit: Qubit):
    """Return a parameterized Rz gate for the specified qubit.

    The gate is a function that takes the angle to rotate and
    returns a LabOne Q section that performs the rotation.
    """

    def rz_gate(angle: float):
        """Rz(theta).

        Theta is in radians - implements a virtual z-gate
        """
        gate = Section(uid=id_generator(f"p_{qubit.uid}_rz_{int(180 * angle / pi)}"))
        gate.play(
            signal=qubit.signals["drive"],
            pulse=None,
            increment_oscillator_phase=angle,
        )

        return gate

    return rz_gate


def measurement(qubit: Qubit):
    """Return a measurement operation of the specified qubit.

    The operation is a function that takes the measurement handle (a string)
    and returns a LabOne Q section that performs the measurement.
    """

    def measurement_gate(handle: str):
        """Perform a measurement.

        Handle is the name of where to store the measurement result. E.g. "meas[0]".
        """
        measure_pulse = pulse_library.const(
            uid=f"{qubit.uid}_readout_pulse",
            length=qubit.parameters.user_defined["readout_len"],
            amplitude=qubit.parameters.user_defined["readout_amp"],
            phase=qubit_parameters['q5']['angle'],
        )

        gate = Section(uid=id_generator(f"meas_{qubit.uid}_{handle}"), trigger={"measure": {"state": True}})
        gate.reserve(signal=qubit.signals["drive"])
        gate.play(signal=qubit.signals["measure"], pulse=measure_pulse)
        gate.acquire(
            signal=qubit.signals["acquire"],
            handle=handle,
            kernel=kernel,
        )
        gate.delay(signal=qubit.signals["measure"], time=120e-6)

        return gate

    return measurement_gate


for oq3_qubit, l1q_qubit in qubit_map.items():
    gate_store.register_gate(
        "sx",
        oq3_qubit,
        drive_pulse_root(l1q_qubit, label="sx"),
        signal=l1q_qubit.signals["drive"],
    )
    gate_store.register_gate(
        "x",
        oq3_qubit,
        drive_pulse(l1q_qubit, label="x"),
        signal=l1q_qubit.signals["drive"],
    )
    gate_store.register_gate_section("rz", (oq3_qubit,), rz(l1q_qubit))
    gate_store.register_gate_section("measure", (oq3_qubit,), measurement(l1q_qubit))

my_session = Session(device_setup=device_setup)
my_session.connect(do_emulation=False)

# %%
qasm_circs = []
n = 100
for i in range(n):
    circuit = QuantumCircuit(1)
    for _ in range(i):
        circuit.sx(0)
    qasm_circs.append(qasm3.dumps(circuit))

exp = exp_from_qasm_list(
    qasm_circs,
    qubits=qubit_map,
    gate_store=gate_store,
    repetition_time=200e-6,
    batch_execution_mode="rt",
    do_reset=False,
    count=1000,
    pipeline_chunk_count=1,
    acquisition_type=acquisition_type,
)

rb1_compiled_exp = my_session.compile(exp)
rabi_results = my_session.run(rb1_compiled_exp)
# %%
acquire_results = rabi_results.acquired_results
vec = abs(acquire_results['measq[0]'].data)

plt.plot(range(n), vec)
plt.plot(range(n), vec, '.')

# plt.ylim([0, 1])
plt.show()
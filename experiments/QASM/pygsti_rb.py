from pygsti.processors import QubitProcessorSpec as QPS
from pygsti.processors import CliffordCompilationRules as CCR
import pygsti
from qiskit import qasm3, transpile

# Define pyGSTi 2 Qubit RB circuit

n_qubits = 2
qubit_labels = ["Q0", "Q1"]
gate_names = ["Gxpi2", "Gxmpi2", "Gypi2", "Gympi2", "Gcphase"]
availability = {"Gcphase": [("Q0", "Q1")]}


pspec = QPS(n_qubits, gate_names, availability=availability, qubit_labels=qubit_labels)

compilations = {
    "absolute": CCR.create_standard(
        pspec, "absolute", ("paulis", "1Qcliffords"), verbosity=0
    ),
    "paulieq": CCR.create_standard(
        pspec, "paulieq", ("1Qcliffords", "allcnots"), verbosity=0
    ),
}

depths = [20, 50]
circuits_per_depth = 2

qubits = ["Q0", "Q1"]

randomizeout = True
citerations = 20

design = pygsti.protocols.CliffordRBDesign(
    pspec,
    compilations,
    depths,
    circuits_per_depth,
    qubit_labels=qubits,
    randomizeout=randomizeout,
    citerations=citerations,
)

circuits_rb = design.all_circuits_needing_data

for circuit in circuits_rb:
    print(circuit)

def sanitize_pygsti_output(
    circuit=circuits_rb,
    pygsti_standard_gates="x-sx-rz",
    qasm_basis_gates=("id", "sx", "x", "rz", "cx"),
    # qasm_basis_gates=("rx","ry","rz","cz"),
):
    qasm2_circuit = []

    for circuit in circuits_rb:
        # pyGSTi standard gates are "u3" and "x-sx-rz""
        qasm2_circuit.append(
            circuit.convert_to_openqasm(standard_gates_version=pygsti_standard_gates)
            .replace("OPENQASM 2.0;", "OPENQASM 3.0;")
            .replace('include "qelib1.inc";', 'include "stdgates.inc";')
            # Support PyGSTI >= 0.9.12 by removing opaque zero delay
            # gates:
            .replace("opaque delay(t) q;", "")
            .replace("delay(0) q[0];", "")
            .replace("delay(0) q[1];", "")
        )

    # qasm3_circuit = []
    #
    # for entry in qasm2_circuit:
    #     qasm3_circuit.append(
    #         qasm3.Exporter().dumps(
    #             transpile(
    #                 qasm3.loads(entry),
    #                 basis_gates=qasm_basis_gates,
    #             )
    #         )
    #     )

    return qasm2_circuit

program_list = sanitize_pygsti_output(
    qasm_basis_gates=["id", "sx", "x", "rz", "cx"],
)


# Import general libraries (needed for functions)
import numpy as np
import matplotlib.pyplot as plt

# Import the RB Functions
from qiskit_ignis_rb import randomized_benchmarking_seq

# Import Qiskit classes
import qiskit
from qiskit import assemble, transpile, qasm3

# Generate RB circuits (2Q RB)

# number of qubits
nQ = 1
rb_opts = {}
# Number of Cliffords in the sequence
rb_opts['length_vector'] = [1, 2, 20, 50, 75, 100, 125, 150, 175, 200]
# Number of seeds (random sequences)
rb_opts['nseeds'] = 5
# Default pattern
rb_opts['rb_pattern'] = [[0]]

rb_opts['is_purity'] = True

a = randomized_benchmarking_seq(**rb_opts)

rb_circs = a[0]
lengths = a[1][0]

circuits = []

for seed_circs in rb_circs:
    for basis_circs in seed_circs:
        for circ in basis_circs:
            circ.remove_final_measurements()
            circuits.append(circ)

rb1_transpiled_circuits = transpile(
    circuits, basis_gates=["id", "sx", "x", "rz"]
)
print(len(circuits))

rb1_program_list = []
for circuit in rb1_transpiled_circuits:
    rb1_program_list.append(qasm3.dumps(circuit))


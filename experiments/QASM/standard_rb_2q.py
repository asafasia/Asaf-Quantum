import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit_experiments.library import randomized_benchmarking
from scipy.optimize import curve_fit
from qubit_parameters import qubit_parameters

from helper.qasm_helper import QuantumProcessor

mode = 'disc'
qubits = ['q1', 'q3']
a = QuantumProcessor(mode=mode, qubits=qubits, pipeline_chunk_count=1, counts=5000)

# %% create qasm circuits

num_samples = 1
lengths = np.arange(1, 5, 1)

rb1_qiskit_circuits = randomized_benchmarking.StandardRB(
    physical_qubits=[1, 0],
    lengths=lengths,
    num_samples=num_samples,
).circuits()

for circuit in rb1_qiskit_circuits:
    circuit.remove_final_measurements()

# Choose basis gates
rb1_transpiled_circuits = transpile(
    rb1_qiskit_circuits, basis_gates=["id", "sx", "x", "rz", "cz"]
)

rb1_program_list = []
for circuit in rb1_transpiled_circuits:
    rb1_program_list.append(qasm3.dumps(circuit))

# %% add to processor and run

a.add_experiment(rb1_program_list)

results = a.run_experiment()

vec_0 = results.acquired_results['measq[0]'].data
vec_1 = results.acquired_results['measq[1]'].data
# vec_2 = results.acquired_results['measq[2]'].data
# vec_3 = results.acquired_results['measq[3]'].data
# vec_4 = results.acquired_results['measq[4]'].data
#
plt.plot(lengths, vec_0, '-o', label='q1')
plt.plot(lengths, vec_1, '-o', label='q3')
# plt.plot(range(n), vec_2, label='q3')
# plt.plot(range(n), vec_3, label='q4')
# plt.plot(range(n), vec_4, label='q5')
plt.ylim([0, 1])
plt.legend()
plt.show()

import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit_experiments.library import randomized_benchmarking
from scipy.optimize import curve_fit

from helper.qasm_helper import QuantumProcessor

mode = 'disc'
a = QuantumProcessor(mode=mode, qubits=['q2'])

# %% create qasm circuits

num_samples = 5
lengths = np.arange(1, 300, 20)

rb1_qiskit_circuits = randomized_benchmarking.StandardRB(
    physical_qubits=[1],
    lengths=lengths,
    num_samples=num_samples,
).circuits()

for circuit in rb1_qiskit_circuits:
    circuit.remove_final_measurements()

# Choose basis gates
rb1_transpiled_circuits = transpile(
    rb1_qiskit_circuits, basis_gates=["id", "sx", "x", "rz"]
)

rb1_program_list = []
for circuit in rb1_transpiled_circuits:
    rb1_program_list.append(qasm3.dumps(circuit))

# %% add to processor and run

a.add_experiment(rb1_program_list)

results = a.run_experiment()

# %% acquiring data

vec = results.acquired_results['measq[0]'].data

matrix = np.reshape(vec, (num_samples, len(lengths)))

probabilities = np.mean(matrix, axis=0)


def f(m, a, b, p):
    return a * p ** m + b


args = curve_fit(f, lengths, probabilities, p0=[0.5, 0.5, 0.99])[0]

plt.plot(lengths, probabilities, 'o-', label='data')
plt.plot(lengths, f(lengths, *args), label=f'fit p = {args[2]:2f} ')

plt.ylim([0, 1])
plt.legend()
plt.show()
p = args[2]
print(f' p = {p}')
print(f' rc = {0.5 * (1 - p):2e}', )

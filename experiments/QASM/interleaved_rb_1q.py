import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit_experiments.library import randomized_benchmarking
from scipy.optimize import curve_fit
from qubit_parameters import qubit_parameters

from helper.qasm_helper import QuantumProcessor

mode = 'disc'
qubit = 'q1'
a = QuantumProcessor(mode=mode, qubits=[qubit])

# %% create qasm circuits

interleaved_element = QuantumCircuit(1)
interleaved_element.x(0)

num_samples = 10
lengths = np.arange(1, 250, 5)

rb1_qiskit_circuits = randomized_benchmarking.InterleavedRB(
    interleaved_element,
    physical_qubits=[0],
    lengths=lengths,
    num_samples=num_samples,
    circuit_order='RRRIII'
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

standard_vec = vec[:len(lengths) * num_samples]
interleaved_vec = vec[len(lengths) * num_samples:]

standard_matrix = np.reshape(standard_vec, (num_samples, len(lengths)))
interleaved_vec = np.reshape(interleaved_vec, (num_samples, len(lengths)))
standard_probabilities = np.mean(standard_matrix, axis=0)
interleaved_probabilities = np.mean(interleaved_vec, axis=0)


def f(m, a, b, p):
    return a * p ** m + b


args = curve_fit(f, lengths, standard_probabilities, p0=[0.5, 0.5, 0.99])[0]

args_interleaved = curve_fit(f, lengths, interleaved_probabilities, p0=[0.5, 0.5, 0.99])[0]

# plt.ylim([0, 1])
plt.legend()
plt.show()
p = args[2]
p_interleaved = args_interleaved[2]
rc = 0.5 * (1 - p)
rc_interleaved = 0.5 * (1 - p_interleaved / p)
print(f' p = {p}')
print(f' p_interleaved = {p_interleaved}')
print(f' rc = {rc:.2e}', )
print(f' rc_interleaved = {rc_interleaved:.2e}', )
plt.xlabel('lengths')
plt.ylabel('probability')
plt.title('Interleaved Randomized Benchmarking: X Gate')
plt.plot(lengths, standard_probabilities, 'o', color='blue', label=f'clifford error =  {rc:.2e}  ')
plt.plot(lengths, f(lengths, *args), color='blue')
plt.plot(lengths, interleaved_probabilities, 'o', color='green', label=f'x gate error = {rc_interleaved:.2e}')
plt.plot(lengths, f(lengths, *args_interleaved), color='green')
plt.legend()
plt.show()
# %% save to labber
# save_labber = True
# if save_labber:
#     import labber.labber_util as lu
#
#     measured_data = dict(population=np.array(probabilities))
#     sweep_parameters = dict(lengths=np.array(lengths))
#     units = dict()
#     meta_data = dict(user="Asaf", tags=[qubit, 'rb'], qubit_parameters=qubit_parameters)
#     exp_result = dict(measured_data=measured_data,
#                       sweep_parameters=sweep_parameters,
#                       units=units,
#                       meta_data=meta_data)
#
#     lu.create_logfile("1q randomized benchmarking", **exp_result, loop_type="1d")

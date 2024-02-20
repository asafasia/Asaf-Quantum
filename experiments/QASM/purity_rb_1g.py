# Import general libraries (needed for functions)
import numpy as np
import matplotlib.pyplot as plt
# Import the RB Functions
from qiskit_ignis_rb import randomized_benchmarking_seq
from qubit_parameters import qubit_parameters
# Import Qiskit classes
from qiskit import assemble, transpile, qasm3
from scipy.optimize import curve_fit

from helper.qasm_helper import QuantumProcessor

# number of qubits
nQ = 1
rb_opts = {}
# Number of Cliffords in the sequence
rb_opts['length_vector'] = np.arange(1, 200, 10)
# Number of seeds (random sequences)
rb_opts['nseeds'] = 8
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

print(rb1_transpiled_circuits[0])
print(rb1_transpiled_circuits[4])
print(rb1_transpiled_circuits[8])

rb1_program_list = []
for circuit in rb1_transpiled_circuits:
    rb1_program_list.append(qasm3.dumps(circuit))

# %% add to processor and run
mode = 'disc'
qubit = 'q1'
a = QuantumProcessor(mode=mode, qubits=[qubit])

a.add_experiment(rb1_program_list)

results = a.run_experiment()

# %% acquiring data
num_samples = rb_opts['nseeds']
vec = results.acquired_results['measqr[0]'].data

new_vec = []
l = len(lengths)
for i in range(len(vec) // 3):
    new_vec.append(np.sqrt((2 * vec[i] - 1) ** 2 + (2 * vec[i + l] - 1) ** 2 + (2 * vec[i + 2 * l] - 1) ** 2))

matrix = np.reshape(new_vec, (num_samples, len(lengths)))

probabilities = np.mean(matrix, axis=0)

plt.plot(lengths, probabilities, 'o')


def f(m, a, b, p):
    return a * p ** (m - 1) + b


args = curve_fit(f, lengths, probabilities, p0=[0.9, 0.1, 0.99])[0]

error_in = 0.5 * (1 - np.sqrt(args[2]))

plt.plot(lengths, f(lengths, *args), label=f'incoherent error = {error_in:.3e}')

plt.xlabel('lengths')
plt.ylabel('probability')
plt.title('Purity RB')
plt.legend()
plt.show()

# %% save labber
save_labber = True
if save_labber:
    import labber.labber_util as lu

    measured_data = dict(population=np.array(probabilities))
    sweep_parameters = dict(lengths=np.array(lengths))
    units = dict()
    meta_data = dict(user="Asaf", tags=[qubit, 'purity rb'], qubit_parameters=qubit_parameters)
    exp_result = dict(measured_data=measured_data,
                      sweep_parameters=sweep_parameters,
                      units=units,
                      meta_data=meta_data)

    lu.create_logfile("1q purity randomized benchmarking", **exp_result, loop_type="1d")

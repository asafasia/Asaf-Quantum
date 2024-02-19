from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit_experiments.library import randomized_benchmarking

from helper.qasm_helper import QuantumProcessor

mode = 'disc'
a = QuantumProcessor(mode=mode)

# %% create qasm circuits
qasm_circs = []
n = 50
for i in range(n):
    circuit = QuantumCircuit(1)
    for _ in range(i):
        circuit.sx(0)
    qasm_circs.append(qasm3.dumps(circuit))


# %% add to processor and run

a.add_experiment(qasm_circs)

results = a.run_experiment()

# %% acquiring data

vec = results.acquired_results['measq[0]'].data

plt.title('X GATES')
plt.plot(range(n), vec)
plt.legend()
plt.ylim([0, 1])
plt.show()

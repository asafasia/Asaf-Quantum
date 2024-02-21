from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, qasm3
from helper.qasm_helper import QuantumProcessor

qubits = ['q1', 'q3']
mode = 'disc'
a = QuantumProcessor(mode=mode, qubits=qubits, pipeline_chunk_count=1,counts=10000)

circs = []
qasm_circs = []
n = 10
for i in range(n):
    circuit = QuantumCircuit(len(qubits))

    circuit.sx(0)
    circuit.x(1)

    for i in range(i):
        circuit.cz(0, 1)

    circuit.sx(0)
    circs.append(circuit)

for circuit in circs:
    qasm_circs.append(qasm3.dumps(circuit))

a.add_experiment(qasm_circs)

results = a.run_experiment()
vec_0 = results.acquired_results['measq[0]'].data
vec_1 = results.acquired_results['measq[1]'].data
# vec_2 = results.acquired_results['measq[2]'].data
# vec_3 = results.acquired_results['measq[3]'].data
# vec_4 = results.acquired_results['measq[4]'].data
#
plt.plot(range(n), vec_0, '-o', label='q1')
plt.plot(range(n), vec_1, '-o', label='q3')
# plt.plot(range(n), vec_2, label='q3')
# plt.plot(range(n), vec_3, label='q4')
# plt.plot(range(n), vec_4, label='q5')
plt.ylim([0, 1])
plt.legend()
plt.show()

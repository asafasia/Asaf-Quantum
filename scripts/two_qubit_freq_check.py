from qubit_parameters import qubit_parameters, update_qp

q1_freq = qubit_parameters["q1"]["qb_freq"]
q2_freq = qubit_parameters["q2"]["qb_freq"]
q3_freq = qubit_parameters["q3"]["qb_freq"]
q4_freq = qubit_parameters["q4"]["qb_freq"]
q5_freq = qubit_parameters["q5"]["qb_freq"]

q1_w125 = qubit_parameters["q1"]["w125"]
q2_w125 = qubit_parameters["q2"]["w125"]
q3_w125 = qubit_parameters["q3"]["w125"]
q4_w125 = qubit_parameters["q4"]["w125"]
q5_w125 = qubit_parameters["q5"]["w125"]

#
q1_anharmonicity = qubit_parameters["q1"]["anharmonicity"]
q3_anharmonicity = qubit_parameters["q3"]["anharmonicity"]
#
#

#
# update_qp(qubit="q3", arg="qb_freq", value=q1_freq - q1_anharmonicity)
#
# print(f"Qubit 2 frequency: {q4_freq}")
# print(f"Qubit 4 frequency: {q4_freq} ")
# print(f"Qubit 5 frequency: {q4_freq}")
#
# print(f"Should Be : {q3_freq - q3_anharmonicity}\n")
#
#

update_qp(qubit="q1", arg="anharmonicity", value=(q1_freq - q1_w125) * 2)
print(f"Qubit 1 frequency: {q1_freq} ")
print(f"Qubit 1 frequency w125: {q1_w125}")
print(f"Qubit1 calculated anharmonicity: {(q1_freq - q1_w125)*2}\n")

print(f"Qubit 1 anharmonicity {q1_anharmonicity} \n")

print(f"Qubit 3 frequency: {q3_freq} ")
print(f"should be : {q1_freq - q1_anharmonicity}\n")

update_qp(qubit="q3", arg="qb_freq", value=(q1_freq - q1_anharmonicity))

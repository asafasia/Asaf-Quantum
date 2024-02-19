import json
from helper import project_path

this_path = project_path + "/qubit_parameters/qp.json"


def load_qp():
    with open(f'{project_path}/qubit_parameters/qp.json', 'r') as file:
        data = json.load(file)
    return data


def load_qp():
    with open(this_path, 'r') as f:
        data = json.load(f)
    return data


def save_qp(data):
    with open(this_path, 'w') as file:
        json.dump(data, file, indent=2)


def update_qp(qubit, arg, value):
    qp = load_qp()
    qp[qubit][arg] = value
    save_qp(qp)


qubit_parameters = load_qp()

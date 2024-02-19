from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from helper.pulses import *
from helper import pulses, exp_helper
from helper.utility_functions import correct_axis
from helper.experiment_results import ExperimentData
from qubit_parameters import qubit_parameters


class QuantumExperiment1Q:
    def __init__(self, q: str, simulate=False, do_emulation=True):
        self.qubit = q
        self.simulate = simulate
        self.qubit_params = qubit_parameters[q]
        self.initializer = exp_helper.initialize_exp()
        self.device_setup = self.initializer.create_device_setup()
        self.exp_signals = self.initializer.signals(q)
        self.signal_map_default = self.initializer.signal_map_default(q)
        self.do_emulation = do_emulation
        self.experiment = None
        self.results = None

    def run(self, ):
        q = self.qubit
        session = Session(device_setup=self.device_setup)
        session.connect(do_emulation=self.do_emulation)

        compiled_qspec = session.compile(self.experiment)
        if self.simulate:
            plot_simulation(compiled_qspec, start_time=0, length=350e-6, signals=[f'drive_{q}', 'measure'])

        self.results = session.run(compiled_qspec)

from laboneq.simple import *
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from laboneq.contrib.example_helpers.plotting.plot_helpers import (
    plot_results,
    plot_simulation,
)

from helper.kernels import kernels
from helper.exp_helper import *
from helper.pulses import *
from qubit_parameters import qubit_parameters, update_qp

from laboneq.contrib.example_helpers.data_analysis.data_analysis import fit_Spec
import scipy.optimize as opt

import json
from datetime import datetime
import os

from collections import OrderedDict

import importlib


# %%


class QubitSpectroscopy:
    def __init__(self, qubit, n_avg, span, amp, steps, w0, center_axis, ground_max, simulate, p, update_flux,
                 mode='spec', ):
        self.qubit = qubit
        self.flux_bias = qubit_parameters[qubit]['flux_bias']
        self.mode = mode

        self.modulation_type = 'hardware' if mode == 'spec' else 'software'
        self.acquisition_type = AcquisitionType.SPECTROSCOPY if mode == 'spec' else AcquisitionType.DISCRIMINATION
        self.kernel = readout_pulse(qubit) if mode == 'spec' else kernels[qubit]

        exp = initialize_exp()

        self.device_setup = exp.create_device_setup(self.modulation_type)
        self.exp_signals = exp.signals(qubit)
        self.signal_map_default = exp.signal_map_default(qubit)

        self.session = Session(device_setup=self.device_setup)
        self.session.connect(do_emulation=False)

        self.update_flux = update_flux
        self.simulate = simulate
        self.n_avg = n_avg
        self.amp = amp  # ~0.4 (q3) for 2nd E level, 1/100 for 1st E level
        self.w0 = w0
        self.center_axis = center_axis
        self.ground_max = ground_max
        self.p = p

        if self.w0:
            self.center = qubit_parameters[qubit]["qb_freq"]
        else:
            self.center = qubit_parameters[qubit]["w125"]

        self.span = span
        self.steps = steps

        drive_LO = qubit_parameters[qubit]["qb_lo"]

        self.dfs = np.linspace(start=self.center - self.span / 2, stop=self.center + self.span / 2,
                               num=self.steps)  # for carrier calculation

        self.freq_sweep = SweepParameter(uid=f'freq_sweep_{qubit}', values=self.dfs - drive_LO)  # sweep object

        self.alpha = (qubit_parameters[qubit]['qb_freq'] - qubit_parameters[qubit]['w125']) * 2

    def _update_vector(self):
        qubit = self.qubit
        self.span = self.span / 1.5
        self.amp = self.amp / 1.5
        drive_LO = qubit_parameters[qubit]["qb_lo"]

        self.dfs = np.linspace(start=self.center - self.span / 2, stop=self.center + self.span / 2,
                               num=self.steps)  # for carrier calculation

        self.freq_sweep = SweepParameter(uid=f'freq_sweep_{qubit}', values=self.dfs - drive_LO)  # sweep object

    def add_experiment(self):

        qubit = self.qubit
        exp_qspec = Experiment(
            uid="Qubit Spectroscopy",
            signals=self.exp_signals,
        )
        with exp_qspec.acquire_loop_rt(
                uid="freq_shots",
                count=self.n_avg,
                acquisition_type=self.acquisition_type,
        ):
            with exp_qspec.sweep(uid="qfreq_sweep", parameter=self.freq_sweep):
                with exp_qspec.section(uid="qubit_excitation"):
                    exp_qspec.play(signal=f"drive_{qubit}", pulse=spec_pulse(self.qubit), amplitude=self.amp,
                                   marker={"marker1": {"enable": True}})

                with exp_qspec.section(
                        uid="readout_section", play_after="qubit_excitation",
                ):
                    # play readout pulse on measure line
                    exp_qspec.play(signal="measure",
                                   pulse=readout_pulse(self.qubit),
                                   phase=qubit_parameters[self.qubit]['angle'],

                                   )

                    # trigger signal data acquisition
                    exp_qspec.acquire(
                        signal="acquire",
                        handle="qubit_spec",
                        kernel=self.kernel,
                        # kernel = kernels[qubit]
                    )
                # delay between consecutive experiment
                with exp_qspec.section(uid="delay"):
                    # relax time after readout - for qubit relaxation to groundstate and signal processing
                    exp_qspec.delay(signal="measure", time=120e-6)

        exp_calibration = Calibration()

        exp_calibration[f"drive_{qubit}"] = SignalCalibration(
            oscillator=Oscillator(
                frequency=self.freq_sweep,
                modulation_type=ModulationType.HARDWARE,
            ),

        )

        exp_qspec.set_calibration(exp_calibration)
        exp_qspec.set_signal_map(self.signal_map_default)

        self.experiment = exp_qspec

    def run_exp(self):
        center = self.center
        compiled_qspec = self.session.compile(self.experiment)

        if self.simulate:
            plot_simulation(compiled_qspec, start_time=0, length=350e-6)

        qspec_results = self.session.run(compiled_qspec)

        results = qspec_results.get_data("qubit_spec")

        amplitude = np.abs(results)
        # amplitude = correct_axis(amplitude,qubit_parameters[qubit]["ge"])

        phase_radians = np.unwrap(np.angle(results))

        drive_amp = qubit_parameters[self.qubit]['drive_amp'] * self.amp

        flux_bias = qubit_parameters[self.qubit]['flux_bias']

        if self.ground_max:
            self.max_freq = self.dfs[np.argmax(amplitude)]
        else:
            self.max_freq = self.dfs[np.argmin(amplitude)]

        amp = self.amp
        qubit = self.qubit
        if self.w0:
            plt.title(
                f'Qubit Spectroscopy {qubit} w0\n drive amp = {amp:5f} V \n flux bias = {self.flux_bias:.5f} V \nresonanance = {center * 1e-6} MHz  ',
                fontsize=18)
        else:
            plt.title(
                f'Qubit Spectroscopy {qubit} w1\n drive amp = {amp:5f} V \n flux bias = {flux_bias:.5f} V \nresonanance = {center * 1e-6:.2f} MHz  ',
                fontsize=18)

        # center*1e-9
        # plt.ylim([-0.2,1.2])
        decimal_places = 1
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter(f"{{x:,.{decimal_places}f}}"))
        # detuning plot
        # plt.plot((dfs - center)*1e-6, amplitude, "k")
        if self.center_axis:
            plt.plot((self.dfs - center) * 1e-6, amplitude, "k")
            plt.axvline(x=0, color='green', linestyle='--', label='current')
            plt.axvline(x=(self.max_freq - center) * 1e-6, color='blue', linestyle='--', label='new')

        else:
            plt.plot(self.dfs * 1e-6, amplitude, "k")
            plt.axvline(x=center * 1e-6, color='green', linestyle='--', label='current')
            plt.axvline(x=self.max_freq * 1e-6, color='blue', linestyle='--', label='new')

        # plt.ylim([0,1])

        # plt.plot(dfs*1e-6 - center*1e-6,func_lorentz(dfs,*popt))
        plt.xlabel('Detuning [MHz]')
        plt.ylabel('Amplitude [a.u.]')
        plt.legend()
        plt.show()

    def update(self):

        detuning = self.max_freq - self.center
        print(f'current detuning: {detuning * 1e-6} [MHz]')
        self._update_vector()

        if self.update_flux and self.w0 == True:
            print('old_flux_point = :', self.flux_bias)

            new_flux_bias = self.flux_bias * (1 + 1e-2 * self.p * detuning * 1e-6)
            print('new_flux_point = :', new_flux_bias)

            user_input = input("Do you want to update the ***flux***? [y/n]")

            if user_input == 'y':
                update_qp(self.qubit, 'flux_bias', new_flux_bias)
                play_flux_qdac(qubit=self.qubit, flux_bias=new_flux_bias)
                self.flux_bias = new_flux_bias
                print('updated !!!')

            elif user_input == 'n':
                print('not_updated')
            else:
                raise Exception("Invalid input")
        else:
            user_input = input("Do you want to update the ***frequency***? [y/n]")

            if self.w0:
                update_string = 'qb_freq'
            else:
                update_string = 'w125'

            if user_input == 'y':
                update_qp(self.qubit, update_string, self.max_freq)

                print('updated !!!')

            elif user_input == 'n':
                print('not_updated')
            else:
                raise Exception("Invalid input")


# %%

if __name__ == '__main__':

    args = {
        'qubit': 'q3',
        'n_avg': 500,
        'simulate': False,
        'amp': 1 / 200,
        'span': 1.5e6,
        'steps': 101,
        'w0': True,
        'center_axis': True,
        'ground_max': False,
        'p': 1,
        'update_flux': False,
        'mode': 'spec'
    }

    # %%

    qs = QubitSpectroscopy(**args)
    # qs.run_exp()
    # qs.update()

    # %%
    n = 1

    for i in range(n):
        print('number: ', i)
        qs.add_experiment()
        qs.run_exp()
        qs.update()



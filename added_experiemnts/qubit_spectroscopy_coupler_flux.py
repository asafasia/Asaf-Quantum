# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 21:18:15 2024

@author: stud
"""


# %% Imports

# LabOne Q:
from laboneq.simple import *
# Helpers:
from laboneq.contrib.example_helpers.plotting.plot_helpers import (
    plot_results,
    plot_simulation,
)
from laboneq.contrib.example_helpers.generate_example_datastore import (
    generate_example_datastore,
    get_first_named_entry,
)
from pathlib import Path
import time
import numpy as np

import matplotlib.pyplot as plt
import math

from helper.exp_helper import *
from helper.kernels import kernels
from helper.pulses import *
from qubit_parameters import qubit_parameters

# %% devise setup

    
    
coupler = 'c43'

if coupler == 'c13':    
    qubit_m = "q1"
    qubit_s = "q3"
    n_port = 0

elif coupler == 'c23':    
    qubit_m = "q3"
    qubit_s = "q2"
    n_port = 1
elif coupler == 'c43':    
    qubit_m = "q3"
    qubit_s = "q4"
    n_port = 2
elif coupler == 'c53':    
    qubit_m = "q3"
    qubit_s = "q5"
    n_port = 3

w0=True
update_flux = True
dip = True
p = 0.1

if w0:
    center = qubit_parameters[qubit_s]["qb_freq_coupler"]
else:
    center = qubit_parameters[qubit_s]["w125"]
    
    
exp = initialize_exp()
device_setup = exp.create_device_setup()
exp_signals = exp.signals(qubit_s)
signal_map_default = exp.signal_map_default(qubit_s)
# %%
session = Session(device_setup=device_setup)
session.connect(do_emulation=False)

# %% parameters for user

# True for pulse simulation plots
simulate = False

# %%parameters for sweeping
# define number of averages
num_averages = 500

# parameters for flux sweep:

coupler_null_flux = qubit_parameters[coupler]['flux_bias']
flux_step_num = 1

# parameters for pulse length sweep:

min_time = 0  # [sec]
max_time = 1e-6  # [sec]
time_step_num =200


# %% parameters for experiment

flux_sweep = LinearSweepParameter(
    "flux_sweep", -0.2, -0.17, flux_step_num)
amp = 1/150
span =2e6
steps = 101
drive_LO = qubit_parameters[qubit_s]["qb_lo"] 

dfs = np.linspace(start=center-span/2, stop=center+span/2, num=steps)  # for carrier calculation

freq_sweep = SweepParameter(uid = f'freq_sweep_{qubit_s}', values = dfs -  drive_LO)  # sweep object

# %%
def qubit_spectroscopy(flux_sweep, freq_sweep):
    # Create qubit spectroscopy Experiment - uses qubit drive, readout drive and data acquisition lines
    
    exp_qspec = Experiment(
        uid="Qubit Spectroscopy",
        signals=exp_signals,     
    )
    
    hdawg_address = device_setup.instrument_by_uid("device_hdawg").address

    with exp_qspec.sweep(uid="flux_sweep", parameter=flux_sweep):
        exp_qspec.set_node(path=f"/{hdawg_address}/sigouts/{n_port}/offset", value=0) 
        
        with exp_qspec.acquire_loop_rt(
            uid="freq_shots",
            count=num_averages,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
            averaging_mode = AveragingMode.CYCLIC 
            
        ):       
            
            with exp_qspec.sweep(uid="qfreq_sweep", parameter=freq_sweep):
                
  
                with exp_qspec.section(uid="flux_section"):
                    exp_qspec.play(
                    signal=f"flux_{coupler}",
                    pulse=flux_pulse(coupler),
                    amplitude=flux_sweep, 
                    length=43e-6)
                    exp_qspec.delay(signal=f"drive_{qubit_s}", time=10e-9)
                    exp_qspec.play(signal=f"drive_{qubit_s}", pulse=spec_pulse(qubit_s), amplitude=amp)

                    
            
                with exp_qspec.section(uid="readout_section",play_after="flux_section"):
                    
                    exp_qspec.play(signal="measure", pulse=readout_pulse(qubit_s))
                    
                    exp_qspec.acquire(
                        signal="acquire",
                        handle="qubit_spec",
                        length=qubit_parameters[qubit_s]["res_len"],
                        
                    )
                # delay between consecutive experiment
                with exp_qspec.section(uid="delay"):
                        # relax time after readout - for qubit relaxation to groundstate and signal processing
                    exp_qspec.delay(signal="measure", time=120e-6)
    return exp_qspec
# %% Run Exp
# define experiment with frequency sweep for qubit 0
exp_qspec = qubit_spectroscopy(flux_sweep, freq_sweep)

exp_calibration = Calibration()


exp_calibration[f"drive_{qubit_s}"] = SignalCalibration(
    oscillator=Oscillator(
        frequency=freq_sweep,
        modulation_type=ModulationType.HARDWARE,
    ),

)


# apply calibration and signal map for qubit 0
exp_qspec.set_signal_map(signal_map_default)
exp_qspec.set_calibration(exp_calibration)

# %% compile exp
# compile the experiment on the open instrument session


if simulate:
    compiled_qspec = session.compile(exp_qspec,compiler_settings={
                                      "SHFSG_MIN_PLAYWAVE_HINT": 256})
    plot_simulation(compiled_qspec, start_time=0, length=300e-6)

# %% run the compiled experiemnt
qspec_results = session.run(exp_qspec)

# %% plot slice
acquire_results = qspec_results.get_data("qubit_spec")

save_func(session,"coupler_spec")

amplitude = np.abs(acquire_results)
#amplitude = correct_axis(amplitude,qubit_parameters[qubit_s]["ge"])

amplitude_T = amplitude.T


x, y = np.meshgrid(flux_sweep.values , dfs - center)

plt.xlabel('Flux [mV]')
plt.ylabel('frequency [us]')
plt.title(f'Qubit Spectroscopy vs. Coupler Flux {qubit_s} {coupler}', fontsize=18)

plt.pcolormesh(x*1e3, y*1e-6, amplitude_T)
# plt.xlim([-220,660])
#plt.axvline(x=coupler_null_flux*1e3, color='red', linestyle='--',label = 'current')
plt.colorbar()
plt.legend()

plt.show()
#%%
if dip:
    max_freq = dfs[np.argmin(amplitude[0,:])]
else:
    max_freq = dfs[np.argmax(amplitude[0,:])]
    


plt.axvline(x=center*1e-6, color='black', linestyle='--',label = 'current resonanace')
plt.axvline(x=max_freq*1e-6, color='red', linestyle='--',label = 'new resonanace')

plt.plot(dfs*1e-6,amplitude[0,:])
plt.legend()
plt.title(f'Qubit Spectroscopy vs. Coupler Flux {qubit_s} {coupler}', fontsize=18)
plt.show()
#%% FFT
# signal = amplitude_T - np.mean(amplitude_T)

# # calculate FFT for each flux value 
# bias_duration=flux_sweep.values
# dt = bias_duration[1] - bias_duration[0]

# bias_level=flux_sweep.values

# bias_freq = np.fft.fftfreq(len(bias_duration), dt)
# bias_freq = np.fft.fftshift(bias_freq)

# Fsignal = np.zeros_like(signal)
# for i in range (len(signal[0, :])):
#     Fsignal[:, i] = abs(np.fft.fft(signal[:, i]))
#     Fsignal[:, i] = np.fft.fftshift(Fsignal[:, i])
    

# normalized = Fsignal/Fsignal.max()
# gamma = 1/4
# corrected = np.power(normalized, gamma) # try values between 0.5 and 2 as a start point

# # plt.axvline(x=565 , color='green', linestyle='--')
# # plt.subplot(1,3,2)
# fig_fft = plt.pcolormesh(bias_level*1e3, abs(bias_freq)*1e-6,corrected, cmap='coolwarm') 
# # plt.xlim([-60,10])
# plt.title(f'FFT Qubit Spectroscopy vs. Coupler Flux  Coupler {coupler}', fontsize=18)

# # type: ignore
# plt.xlabel('coupler bias [mV]')
# plt.ylabel('bias frequency [MHz]') 
# plt.colorbar()
# plt.show()

# %%
flux_bias = qubit_parameters[qubit_s]['flux_bias']
detuning = max_freq - center
print(f'current detuning: {detuning*1e-6} [MHz]' )



if update_flux and w0==True:
    print('old_flux_point = :', flux_bias)


    new_flux_bias = flux_bias*(1 + 1e-2*p*detuning*1e-6)
    print('new_flux_point = :', new_flux_bias)


    user_input = input("Do you want to update the ***flux***? [y/n]")

    if user_input == 'y':
        update_qp(qubit_s,'flux_bias' ,new_flux_bias)
        print('updated !!!')
    
    elif  user_input == 'n':
        print('not_updated')
    else:
        raise Exception("Invalid input")
else:
    user_input = input("Do you want to update the ***frequency***? [y/n]")


    if w0:
        update_string = 'qb_freq'
    else:
        update_string = 'w125'
        
        
    if user_input == 'y':
        update_qp(qubit_s,update_string ,max_freq)
        
        print('updated !!!')
    
    elif  user_input == 'n':
        print('not_updated')
    else:
        raise Exception("Invalid input")

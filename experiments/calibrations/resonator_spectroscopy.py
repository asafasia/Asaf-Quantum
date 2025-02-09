import matplotlib.pyplot as plt
from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
from helper.exp_helper import *
from helper.pulses import *
from qubit_parameters import qubit_parameters, update_qp
import json
from datetime import datetime
import os
from collections import OrderedDict

# %% devise setup

qubit = "q3"

exp = initialize_exp()
device_setup = exp.create_device_setup()
signal_map_default = exp.signal_map_default(qubit)
exp_signals = exp.signals(qubit)

# %%

session = Session(device_setup=device_setup, )
session.connect(do_emulation=False)

# %% parameters
long_pulse = False
simulate = False
plot_from_json = False
num_averages = 300

freq_range = 200e6  # for resonator spectroscopy drive sweep
steps = 101  # odd num - number of steps in frequency sweep

central_frequency = qubit_parameters[qubit]["res_freq"]

CF_freq_vec = np.linspace(start=central_frequency - freq_range / 2,
                          stop=central_frequency + freq_range / 2, num=steps)  # for carrier calculation

res_LO = qubit_parameters[qubit]["res_lo"]

freq_sweep_q0 = SweepParameter(uid='freq_sweep', values=CF_freq_vec - res_LO)  # sweep object


# %% pulse parameters and definiations
# envelope_duration = 2e-6
# sigma = 0.2
# flat_duration = 1.0e-6

# %% experiment


def qubit_spectroscopy(freq_sweep):
    # Create qubit spectroscopy Experiment - uses qubit drive, readout drive and data acquisition lines
    exp_qspec = Experiment(
        uid="Qubit Spectroscopy",
        signals=exp_signals,

    )
    # inner loop - real-time averaging - QA in integration mode
    with exp_qspec.acquire_loop_rt(
            uid="freq_shots",
            count=num_averages,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
    ):
        for i in range(2):
            with exp_qspec.sweep(parameter=freq_sweep):
                # qubit drive
                with exp_qspec.section():
                    if i == 1:
                        if long_pulse:
                            exp_qspec.play(signal=f"drive_{qubit}", pulse=spec_pulse(qubit))
                        else:
                            exp_qspec.play(signal=f"drive_{qubit}", pulse=pi_pulse(qubit))

                        # else:

                        exp_qspec.delay(signal=f"drive_{qubit}", time=0e-9)

                with exp_qspec.section(trigger={"measure": {"state": True}}):
                    exp_qspec.reserve(f"drive_{qubit}")
                    # play readout pulse on measure line
                    exp_qspec.play(signal="measure",
                                   pulse=readout_pulse(qubit),
                                   phase=qubit_parameters[qubit]['angle'])
                    # trigger signal data acquisition
                    exp_qspec.acquire(
                        signal="acquire",
                        handle=f"spec_{i + 1}",
                        length=qubit_parameters[qubit]["res_len"],
                    )
                with exp_qspec.section():  # delay between consecutive experiment
                    # relax time after readout - for qubit relaxation to groundstate and signal processing
                    exp_qspec.delay(signal="measure", time=120e-6)
    return exp_qspec


# %% frequency range of spectroscopy scan - defined around expected qubit frequency as defined in qubit parameters


exp_calibration = Calibration()

exp_calibration["measure"] = SignalCalibration(
    oscillator=Oscillator(
        "readout_osc",
        frequency=freq_sweep_q0,
        modulation_type=ModulationType.HARDWARE,
    ),
)

exp_qspec = qubit_spectroscopy(freq_sweep_q0)
exp_qspec.set_calibration(exp_calibration)
exp_qspec.set_signal_map(signal_map_default)

# %% compile

compiled_qspec = session.compile(exp_qspec)
if simulate:
    plot_simulation(compiled_qspec, )
# %% plot

res_spec = session.run(compiled_qspec)

# %%
acquire_results_1 = res_spec.get_data("spec_1")

acquire_results_2 = res_spec.get_data("spec_2")
# %% save results

# import json
# from datetime import datetime
# import os

# from collections import OrderedDict

# experiment_name="resonator_spectroscopy"
# timestamp = time.strftime("%Y%m%dT%H%M%S")
# Path("Results").mkdir(parents=True, exist_ok=True)
# session.save_results(f"Results/{timestamp}_qb_flux_results_{qubit}.json")
# print(f"File saved as Results/{timestamp}__qb_flux_results_{qubit}.json.json")

# current_date = datetime.now().strftime("%Y-%m-%d")
# # Create a folder based on the current date if it doesn't exist
# folder_path = os.path.join("C:/Users/stud/Documents/GitHub/qhipu-files/LabOne Q/Exp_results", f"{current_date}_results")
# if not os.path.exists(folder_path):
#     os.makedirs(folder_path)

# # Generate a unique filename based on the current time
# current_time = datetime.now().strftime("Time_%H-%M-%S")
# filename = os.path.join(folder_path, f"{experiment_name}_{current_time}.json")

# session.save_results(filename)

# print(f"Data saved to {filename}")

# %% set plotting variables
# **** don't forget to update file path

amplitude_1 = np.abs(acquire_results_1)
amplitude_2 = np.abs(acquire_results_2)

phase_radians_1 = np.unwrap(np.angle(acquire_results_1)) * amplitude_1
phase_radians_2 = np.unwrap(np.angle(acquire_results_2)) * amplitude_2

# Convert res_freq to GHz
res_freq_GHz = CF_freq_vec * 1e-9
# save results to json file


# %%
diff = abs(amplitude_2 - amplitude_1)

max_freq = res_freq_GHz[np.argmax(diff)]
print(max_freq)

# %% plot

# Create a figure with 2 subplots stacked vertically
fig, axes = plt.subplots(3, 1, figsize=(8, 6))
drive_amp = qubit_parameters[qubit]['drive_amp']
res_amp = qubit_parameters[qubit]['res_amp']
res_len = qubit_parameters[qubit]['res_len']

fig.suptitle(
    f'Resonator Spectroscopy {qubit} \n drive amp = {drive_amp} V \n res amp = {res_amp:.2f} V \n res len = {res_len * 1e6} us',
    fontsize=18)

# Plot the amplitude in the first subplot
axes[0].plot(res_freq_GHz, amplitude_1, color='blue', marker='.', label='with_drive')
axes[0].plot(res_freq_GHz, amplitude_2, color='green', marker='.', label='without_drive')
axes[2].set_xlabel('Frequency [GHz]')

axes[0].set_xlabel('Frequency [GHz]')
axes[0].set_ylabel('Amplitude [a.u.]')
axes[0].grid(True)

# Plot the phase in radians in the second subplot
axes[1].plot(res_freq_GHz, phase_radians_1, color='red', marker='.', label='with_drive')
axes[1].plot(res_freq_GHz, phase_radians_2, color='yellow', marker='.', label='without_drive')
axes[1].set_xlabel('Frequency [GHz]')
axes[1].set_ylabel('Phase [rad]')
axes[1].grid(True)
axes[0].axvline(x=qubit_parameters[qubit]["res_freq"] * 1e-9, color='black', linestyle='--')
axes[1].axvline(x=qubit_parameters[qubit]["res_freq"] * 1e-9, color='black', linestyle='--')
axes[2].axvline(x=qubit_parameters[qubit]["res_freq"] * 1e-9, color='black', linestyle='--', label='current resonanace')

axes[0].axvline(x=max_freq, color='blue', linestyle='--')
axes[1].axvline(x=max_freq, color='blue', linestyle='--')
axes[2].axvline(x=max_freq, color='blue', linestyle='--', label='new resonance')

# qubit_parameters[qubit]["res_freq"]*1e-9
axes[2].plot(res_freq_GHz, abs(amplitude_2 - amplitude_1), color='red', marker='.', label='diff')

axes[1].legend()
axes[0].legend()
axes[2].legend()

plt.tight_layout()
plt.show()

GateStore()

# %% update
user_input = input("Do you want to update the pi amplitude? [y/n]")

if user_input == 'y':
    update_qp(qubit, 'res_freq', max_freq * 1e9)
    print('updated !!!')

elif user_input == 'n':
    print('not_updated')
else:
    raise Exception("Invalid input")

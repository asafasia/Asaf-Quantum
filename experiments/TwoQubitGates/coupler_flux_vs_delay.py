from helper.pulses import readout_pulse
import matplotlib.pyplot as plt
from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
from laboneq.simple import *
import numpy as np
from helper import pulses, exp_helper
from qubit_parameters import qubit_parameters
from helper.kernels import kernels

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
else:
    raise ValueError('Coupler not found')

mode = 'disc'
modulation_type = 'hardware' if mode == 'spec' else 'software'
if mode == 'spec':
    acquisition_type = AcquisitionType.SPECTROSCOPY
elif mode == 'int':
    acquisition_type = AcquisitionType.INTEGRATION
elif mode == 'disc':
    acquisition_type = AcquisitionType.DISCRIMINATION

kernel = readout_pulse(qubit_s) if mode == 'spec' else kernels[qubit_s]

exp = exp_helper.initialize_exp()
device_setup = exp.create_device_setup(modulation_type=modulation_type)
exp_signals = exp.signals(qubit_s)
signal_map_default = exp.signal_map_default(qubit_s)

# %%
session = Session(device_setup=device_setup)
session.connect(do_emulation=False)

# %% parameters for user
central_frequency = qubit_parameters[qubit_s]["qb_freq"]
simulate = False

# %%parameters for sweeping
exp_repetitions = 200

# parameters for flux sweep:
min_flux = 30e-3  # [V]
max_flux = -100e-3  # [V]
flux_step_num = 22

# parameters for pulse length sweep:
min_time = 0  # [sec]
max_time = 0.4e-6  # [sec]
time_step_num = 10

# %% parameters for experiment

flux_sweep = LinearSweepParameter(
    "flux_sweep", min_flux, max_flux, flux_step_num)
time_sweep = LinearSweepParameter(
    "freq_RF_sweep", min_time, max_time, time_step_num)


# %%
def qubit_spectroscopy(flux_sweep, time_sweep):
    # Create qubit spectroscopy Experiment - uses qubit drive, readout drive and data acquisition lines
    exp_qspec = Experiment(
        uid="Qubit Spectroscopy",
        signals=exp_signals

    )
    hdawg_address = device_setup.instrument_by_uid("device_hdawg").address

    with exp_qspec.sweep(uid="near-time sweep", parameter=flux_sweep, execution_type=ExecutionType.NEAR_TIME):
        exp_qspec.set_node(path=f"/{hdawg_address}/sigouts/{n_port}/offset", value=0)
        with exp_qspec.acquire_loop_rt(
                uid="freq_shots",
                count=exp_repetitions,
                acquisition_type=acquisition_type,
                averaging_mode=AveragingMode.CYCLIC

        ):
            with exp_qspec.sweep(uid="qfreq_sweep", parameter=time_sweep):
                with exp_qspec.section(uid="qubit_excitation"):
                    exp_qspec.play(signal=f"drive_{qubit_m}", pulse=pulses.pi_pulse(qubit_m))
                    exp_qspec.play(signal=f"drive_{qubit_s}", pulse=pulses.pi_pulse(qubit_s))

                with exp_qspec.section(uid="time_delay", play_after="qubit_excitation"):
                    exp_qspec.delay(signal="measure", time=time_sweep)

                with exp_qspec.section(uid="readout_section", play_after="time_delay"):
                    exp_qspec.play(signal="measure", pulse=pulses.readout_pulse(qubit_s))

                    exp_qspec.acquire(
                        signal="acquire",
                        handle="qubit_spec",
                        kernel=kernel,
                    )
                with exp_qspec.section(uid="delay"):
                    exp_qspec.delay(signal="measure", time=120e-6)
    return exp_qspec


# %% Run Exp
exp_qspec = qubit_spectroscopy(flux_sweep, time_sweep)
exp_qspec.set_signal_map(signal_map_default)

# %% compile exp
compiled_qspec = session.compile(exp_qspec)
if simulate:
    plot_simulation(compiled_qspec, start_time=0, length=300e-6)

# %% run the compiled experiment
qspec_results = session.run(exp_qspec)

# %%
acquire_results = qspec_results.get_data("qubit_spec")
amplitude = np.real(acquire_results)

# %%
# index = 0
# print(time_sweep.values.shape)
# print('Current flux: ', flux_sweep.values[index])
#
# plt.title(f'CPhase oscillation vs. time \nflux point = {flux_sweep.values[index]} V')
# plt.plot(time_sweep.values * 1e6, amplitude[index, :])
# plt.ylabel('Amplitude [a.u.]')
# plt.xlabel('time [us]')
# plt.show()

# %% plot 2d

amplitude = amplitude.T

x, y = np.meshgrid(flux_sweep.values, time_sweep.values)

plt.xlabel('Flux [mV]')
plt.ylabel('time [us]')
plt.title(f'Qubit Spectroscopy vs. Coupler Flux {qubit_s} {coupler}', fontsize=18)

plt.pcolormesh(x * 1e3, y * 1e6, amplitude)

plt.colorbar()
plt.show()

# %% FFT
signal = amplitude - np.mean(amplitude)

# calculate FFT for each flux value
bias_duration = time_sweep.values
dt = bias_duration[1] - bias_duration[0]

bias_level = flux_sweep.values

bias_freq = np.fft.fftfreq(len(bias_duration), dt)
bias_freq = np.fft.fftshift(bias_freq)

Fsignal = np.zeros_like(signal)
for i in range(len(signal[0, :])):
    Fsignal[:, i] = abs(np.fft.fft(signal[:, i]))
    Fsignal[:, i] = np.fft.fftshift(Fsignal[:, i])

normalized = Fsignal / Fsignal.max()
gamma = 1 / 4
corrected = np.power(normalized, gamma)  # try values between 0.5 and 2 as a start point

# plt.axvline(x=565, color='green', linestyle='--')
fig_fft = plt.pcolormesh(bias_level * 1e3, abs(bias_freq) * 1e-6, corrected, cmap='coolwarm')
plt.title(f'FFT Qubit Spectroscopy vs. Coupler Flux  Coupler {coupler}', fontsize=18)

plt.xlabel('coupler bias [mV]')
plt.ylabel('bias frequency [MHz]')
plt.colorbar()
plt.show()

# %% save to labber

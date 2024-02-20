from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
import matplotlib.pyplot as plt
from helper.exp_helper import *
from helper.pulses import *
from helper.kernels import kernels
from qubit_parameters import qubit_parameters
from scipy.optimize import curve_fit

# %% devise setup
coupler = 'c13'

if coupler == 'c13':
    qubit_m = "q1"
    qubit_s = "q3"
    n_port = 0

elif coupler == 'c23':
    qubit_m = "q3"
    qubit_s = "q2"
    n_port = 1
elif coupler == 'c43':
    qubit_s = "q3"
    qubit_m = "q4"
    n_port = 2
elif coupler == 'c53':
    qubit_s = "q3"
    qubit_m = "q5"
    n_port = 3

mode = 'disc'
modulation_type = 'hardware' if mode == 'spec' else 'software'
if mode == 'spec':
    acquisition_type = AcquisitionType.SPECTROSCOPY
elif mode == 'int':
    acquisition_type = AcquisitionType.INTEGRATION
elif mode == 'disc':
    acquisition_type = AcquisitionType.DISCRIMINATION

kernel = readout_pulse(qubit_m) if mode == 'spec' else kernels[qubit_s]

exp = initialize_exp()
device_setup = exp.create_device_setup(modulation_type)
exp_signals = exp.signals(qubit_s)
signal_map_default = exp.signal_map_default(qubit_s)
# %%
session = Session(device_setup=device_setup)
session.connect(do_emulation=True)

# %% parameters for user
central_frequency = qubit_parameters[qubit_s]["qb_freq"]
# True for pulse simulation plots
simulate = True
plot_from_json = False
# %%parameters for sweeping
num_averages = 100
flux = -0.02
max_time = 0.5e-6  # [sec]
time_step_num = 20

# %% parameters for experiment
time_sweep = LinearSweepParameter(
    "freq_sweep", 0, max_time, time_step_num)


# %%
def qubit_spectroscopy(time_sweep):
    # Create qubit spectroscopy Experiment - uses qubit drive, readout drive and data acquisition lines
    exp_qspec = Experiment(
        uid="Qubit Spectroscopy",
        signals=exp_signals,
    )

    with exp_qspec.acquire_loop_rt(
            uid="freq_shots",
            count=num_averages,
            acquisition_type=acquisition_type,
            averaging_mode=AveragingMode.CYCLIC

    ):
        # sweep qubit drive

        with exp_qspec.sweep(uid="qfreq_sweep", parameter=time_sweep):
            with exp_qspec.section(uid="qubit_excitation"):
                exp_qspec.play(signal=f"drive_{qubit_s}", pulse=pulse_library.const(
                    uid=f"pi_pulse_1",
                    length=qubit_parameters[qubit_s]["pi_len"],
                    amplitude=qubit_parameters[qubit_s]["pi_amp"] * 1 / 2,
                    can_compress=False
                ))
            # exp_qspec.play(signal=f"drive_{qubit_m}", pulse=pi_pulse(qubit_m), amplitude=1/2)

            with exp_qspec.section(uid="flux_section", play_after='qubit_excitation'):
                exp_qspec.play(
                    signal=f"flux_{coupler}",
                    pulse=flux_pulse(coupler),
                    length=100e-9,
                    amplitude=-1)

            with exp_qspec.section(uid="time_delay", play_after="flux_section"):
                exp_qspec.play(signal=f"drive_{qubit_s}", pulse=pi_pulse(qubit_s), amplitude=1 / 2,
                               phase=-2 * np.pi * 0 * time_sweep)
                # exp_qspec.delay(signal="measure", time=10e-9)

            with exp_qspec.section(uid="readout_section", play_after="time_delay"):
                exp_qspec.play(signal="measure", pulse=readout_pulse(qubit_s))

                exp_qspec.acquire(
                    signal="acquire",
                    handle="qubit_spec",
                    kernel=kernel
                )
            # delay between consecutive experiment
            with exp_qspec.section(uid="delay"):
                # relax time after readout - for qubit relaxation to groundstate and signal processing
                exp_qspec.delay(signal="measure", time=1e-6)
        return exp_qspec

# %% Run Exp
exp_qspec = qubit_spectroscopy(time_sweep)
exp_qspec.set_signal_map(signal_map_default)
# %% compile exp
compiled_qspec = session.compile(exp_qspec)

if simulate:
    plot_simulation(compiled_qspec, start_time=0, length=1e-6, signals=[f'flux_{coupler}', f'drive_{qubit_s}'])
plt.show()
# %% run the compiled experiemnt
qspec_results = session.run(exp_qspec)
acquire_results = qspec_results.get_data("qubit_spec")
amplitude = np.abs(acquire_results)

# %%

plt.plot(time_sweep, amplitude)

def cos_wave(x, amplitude, T, offset):
    return amplitude * np.cos(2 * np.pi / T * x) + offset

# Assuming your signal is e^(i * theta)
# phase_guess = np.angle(np.mean(amplitude))  # Initial guess for the angle

# guess = [(max(amplitude) - min(amplitude)) * 10, 1e-7, np.mean(amplitude)]

# params, params_covariance = curve_fit(cos_wave, time_sweep, amplitude, p0=guess)

# Plot the amplitude in the first subplot
# plt.plot(time_sweep, cos_wave(time_sweep, *params),
#          label='Fitted function', color='red')
plt.xlabel('Drive Amplitude [a.u.]')
plt.ylabel('Amplitude [a.u.]')
plt.legend()
plt.title(f'Phase Oscillations {qubit_s}', fontsize=18)
# phase_shift = 2 * np.pi / params[1] * time_sweep
# phase_shift_parameter = params[1]
# print("phase_shift_helper =", params[1])

plt.xlabel('time [us]')
plt.ylabel('amp [us]')
plt.title(f'Cz Oscillations vs. Coupler Flux {qubit_s} {coupler}', fontsize=18)
# plt.show()

from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
import matplotlib.pyplot as plt
from helper.kernels import kernels
from helper.exp_helper import *
from helper.pulses import *
from qubit_parameters import qubit_parameters, update_qp
from scipy.optimize import curve_fit

# %%
coupler = 'c13'

if coupler == 'c13':
    qubit_m = "q1"
    qubit_s = "q3"
    n_port = 5

elif coupler == 'c23':
    qubit_m = "q3"
    qubit_s = "q2"
    n_port = 6
elif coupler == 'c43':
    qubit_m = "q3"
    qubit_s = "q4"
    n_port = 7

mode = 'disc'
modulation_type = 'hardware' if mode == 'spec' else 'software'
if mode == 'spec':
    acquisition_type = AcquisitionType.SPECTROSCOPY
elif mode == 'int':
    acquisition_type = AcquisitionType.INTEGRATION
elif mode == 'disc':
    acquisition_type = AcquisitionType.DISCRIMINATION

kernel = readout_pulse(qubit_s) if mode == 'spec' else kernels[qubit_s]

exp = initialize_exp()
device_setup = exp.create_device_setup(modulation_type)
exp_signals = exp.signals(qubit_s)
signal_map_default = exp.signal_map_default(qubit_s)
# %%
session = Session(device_setup=device_setup)
session.connect(do_emulation=False)

# %% parameters for user

# central frequency for drive frequency sweep:
central_frequency = qubit_parameters[qubit_s]["qb_freq"]
phase = qubit_parameters[qubit_s]["cz_phase_shift_T"]
# True for pulse simulation plots
simulate = False
plot_from_json = False
# %%parameters for sweeping
# define number of averages
num_averages = 500

# parameters for flux sweep:

flux = -0.036

# parameters for pulse length sweep:

max_time = 0.2e-6  # [sec]
time_step_num = 200

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
                    amplitude=flux,
                    length=time_sweep)

            with exp_qspec.section(uid="time_delay", play_after="flux_section"):
                exp_qspec.play(signal=f"drive_{qubit_s}", pulse=pi_pulse(qubit_s), amplitude=1 / 2,
                               phase=0 * phase * time_sweep)
                # exp_qspec.delay(signal="measure", time=10e-9)

            with exp_qspec.section(uid="readout_section", play_after="time_delay"):
                exp_qspec.play(signal="measure", pulse=readout_pulse(qubit_s), phase=qubit_parameters[qubit_s]['angle'])

                exp_qspec.acquire(
                    signal="acquire",
                    handle="qubit_spec",
                    kernel=kernel
                )
            with exp_qspec.section(uid="delay"):
                exp_qspec.delay(signal="measure", time=120e-6)
    return exp_qspec


# %% Run Exp
exp_qspec = qubit_spectroscopy(time_sweep)
exp_qspec.set_signal_map(signal_map_default)
# %% compile exp
if simulate:
    compiled_qspec = session.compile(exp_qspec)
    plot_simulation(compiled_qspec, start_time=0, length=1e-6, signals=[f'flux_{coupler}', f'drive_{qubit_s}'])

# %% run the compiled experiemnt
qspec_results = session.run(exp_qspec)
acquire_results = qspec_results.get_data("qubit_spec")
amplitude = np.abs(acquire_results)

time_sweep = time_sweep.values


# %% plot slice

def cos_wave(x, amplitude, T, offset):
    return amplitude * np.cos(2 * np.pi / T * x) + offset


# Assuming your signal is e^(i * theta)
phase_guess = np.angle(np.mean(amplitude))  # Initial guess for the angle

guess = [(max(amplitude) - min(amplitude)) * 10, 0.125e-6, np.mean(amplitude)]

params, params_covariance = curve_fit(cos_wave, time_sweep, amplitude, p0=guess)

plt.plot(time_sweep, amplitude, label='Data', color='blue')
# Plot the amplitude in the first subplot
plt.plot(time_sweep, cos_wave(time_sweep, *params),
         label='Fitted function', color='red')
plt.xlabel('Drive Amplitude [a.u.]')
plt.ylabel('Amplitude [a.u.]')
plt.legend()
# Set a single big title for the entire figure
plt.title(f'Phase Oscillations {qubit_s}', fontsize=18)

phase_shift = 2 * np.pi / params[1] * time_sweep
T_phase_shift_helper = params[1] * 2 * np.pi

update_qp(qubit_s, 'cz_phase_shift_T', T_phase_shift_helper)

print("phase = ", params[1])

plt.xlabel('time [us]')
plt.ylabel('amp [us]')

plt.title(f'Cz Oscillations vs. Coupler Flux {qubit_s} {coupler}', fontsize=18)

plt.tight_layout()
plt.show()

from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
import matplotlib.pyplot as plt
from helper.kernels import kernels
from helper.exp_helper import *
from helper.pulses import *
from qubit_parameters import qubit_parameters
from scipy.optimize import curve_fit

# %%

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
    qubit_m = "q3"
    qubit_s = "q4"
    n_port = 2

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

central_frequency = qubit_parameters[qubit_s]["qb_freq"]
simulate = False
plot_from_json = False

# %%parameters for sweeping
num_averages = 500
flux = qubit_parameters[coupler]["coupling_flux"] - qubit_parameters[coupler]["flux_bias"]
max_time = 50e-9  # [sec]
time_step_num = 200

# %% parameters for experiment
time_sweep = LinearSweepParameter(
    "freq_sweep", 0, max_time, time_step_num)

# %%
def qubit_spectroscopy(time_sweep):
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
                    amplitude=qubit_parameters[qubit_s]["pi_amp"],
                    can_compress=False
                ))
                exp_qspec.play(signal=f"drive_{qubit_m}", pulse=pi_pulse(qubit_m))

            with exp_qspec.section(uid="flux_section", play_after='qubit_excitation'):
                exp_qspec.play(
                    signal=f"flux_{coupler}",
                    pulse=flux_pulse(coupler),
                    amplitude=flux,
                    length=time_sweep)

            with exp_qspec.section(uid="readout_section", play_after="flux_section"):
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
    plot_simulation(compiled_qspec, start_time=0, length=300e-6)

# %% run the compiled experiment
qspec_results = session.run(exp_qspec)
acquire_results = qspec_results.get_data("qubit_spec")
amplitude = np.abs(acquire_results)
time_sweep = time_sweep.values

# %% plot slice
def cos_wave(x, amplitude, T, phase, offset):
    return amplitude * np.sin(2 * np.pi / T * x + phase) + offset

guess = [(max(amplitude) - min(amplitude)), 42e-9, 0, np.mean(amplitude)]
params, params_covariance = curve_fit(cos_wave, time_sweep, amplitude, p0=guess)
plt.plot(time_sweep / 1e-9, amplitude)
plt.plot(time_sweep / 1e-9, cos_wave(time_sweep, *params),
         label='Fitted function', color='red')

plt.xlabel('time [ns]')
plt.ylabel('population [a.u]')
plt.title(f'Cz Oscillation {qubit_s} {coupler}', fontsize=18)
plt.tight_layout()
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import cmath
from helper.pulses import *
from helper import pulses, exp_helper
from helper.experiment_results import ExperimentData
from helper.plotter import Plotter
from qubit_parameters import update_qp, qubit_parameters

# %% devise setup
qubit = "q4"
exp = exp_helper.initialize_exp()
device_setup = exp.create_device_setup(modulation_type='hardware')
exp_signals = exp.signals(qubit)
signal_map_default = exp.signal_map_default(qubit)

# %% parameters
do_emulation = False
simulate = True
show_clouds = True
exp_repetitions = 20000

# %% experiment
exp_0 = Experiment(
    uid="Optimal weights",
    signals=[
        ExperimentSignal("measure"),
        ExperimentSignal("acquire"),
        ExperimentSignal(f"drive_{qubit}"),
    ],
)

with exp_0.acquire_loop_rt(
        uid="shots",
        count=exp_repetitions,  # pow(2, average_exponent),
        averaging_mode=AveragingMode.SINGLE_SHOT,
        acquisition_type=AcquisitionType.SPECTROSCOPY,

):
    with exp_0.section():
        exp_0.play(signal=f"drive_{qubit}", pulse=pi_pulse(qubit), amplitude=0)

    with exp_0.section():
        exp_0.play(signal="measure",
                   pulse=readout_pulse(qubit),
                   phase=qubit_parameters[qubit]['angle'])
        exp_0.acquire(
            signal="acquire", handle="ac_0", length=readout_pulse(qubit).length,
        )

    with exp_0.section():
        exp_0.delay(signal="measure", time=120e-6)

exp_1 = Experiment(
    uid="Optimal weights",
    signals=[
        ExperimentSignal("measure"),
        ExperimentSignal("acquire"),
        ExperimentSignal(f"drive_{qubit}"),
    ],
)

with exp_1.acquire_loop_rt(
        uid="shots",
        count=exp_repetitions,
        averaging_mode=AveragingMode.SINGLE_SHOT,
        acquisition_type=AcquisitionType.SPECTROSCOPY,
):
    with exp_1.section(uid='drive'):
        exp_1.play(signal=f"drive_{qubit}", pulse=pi_pulse(qubit), amplitude=1)

    with exp_1.section(uid='measure'):
        exp_1.reserve(signal=f"drive_{qubit}")

        exp_1.play(signal="measure",
                   pulse=pulses.readout_pulse(qubit),
                   phase=qubit_parameters[qubit]['angle']
                   )

        exp_1.acquire(
            signal="acquire",
            handle="ac_1",
            length=pulses.readout_pulse(qubit).length,
        )

    with exp_1.section():
        exp_1.delay(signal="measure", time=120e-6)

signal_map_default = {f"drive_{qubit}": device_setup.logical_signal_groups[qubit].logical_signals["drive_line"],
                      "measure": device_setup.logical_signal_groups[qubit].logical_signals["measure_line"],
                      "acquire": device_setup.logical_signal_groups[qubit].logical_signals["acquire_line"]}

exp_0.set_signal_map(signal_map_default)
exp_1.set_signal_map(signal_map_default)

# %% run
session = Session(device_setup=device_setup)
session.connect(do_emulation=do_emulation)

results_0 = session.run(exp_0)
results_1 = session.run(exp_1)

# %% acquire data
raw_0 = np.array(results_0.get_data("ac_0"))
raw_1 = np.array(results_1.get_data("ac_1"))

traces = np.array([raw_0, raw_1])

m1 = np.mean(traces[0])
m2 = np.mean(traces[1])
v1 = np.var(traces[0])
v2 = np.var(traces[1])

angle = -cmath.phase(m2 - m1)
separation = abs((m2 - m1) / np.sqrt(v1))
# print(f"ground = {m1.real} \nexceited = {m2.real}")
# print(f"angle = {angle + qubit_parameters[qubit]['angle']}")

if not do_emulation:
    print('saved!!!!')
    update_qp(qubit, 'angle', angle + qubit_parameters[qubit]['angle'])
    update_qp(qubit, 'ge', [m1.real, m2.real])

# %% clouds
if show_clouds:
    plt.plot(traces[0].real, traces[0].imag, '.', alpha=0.6, label='ground')
    plt.plot(traces[1].real, traces[1].imag, '.', alpha=0.6, label='excited')

    # mean
    plt.plot(m1.real, m1.imag, 'x', color='black')
    plt.plot(m2.real, m2.imag, 'x', color='black')

    plt.title(f'IQ Clouds {qubit}')
    plt.xlabel('I [a.u.]', )
    plt.ylabel('Q [a.u.]')
    plt.legend()

    plt.show()

# %% histogram
res_freq = qubit_parameters[qubit]['res_freq']
res_amp = qubit_parameters[qubit]['res_amp']
res_len = qubit_parameters[qubit]['res_len']

plt.title(
    f"Seperation Histogram {qubit} \nSeperation = {separation:.3f} \nres_freq = {res_freq * 1e-6:.2f} MHz  \nres amp = {res_amp} V \nres len = {res_len * 1e6} us ")
plt.hist(traces[0].real, bins=80, edgecolor='black', label='ground state', alpha=0.6)
plt.hist(traces[1].real, bins=80, edgecolor='black', label='excited state', alpha=0.5)

plt.xlabel('I [a.u.]')
plt.ylabel('number')
plt.legend()
plt.axvline(x=m2 / 2 + m1 / 2, linestyle='--', color='black')
plt.axvline(x=m1, linestyle='--', color='blue')
plt.axvline(x=m2, linestyle='--', color='red')

plt.show()

print(f'separation = {separation:.3f}')

# %% save results
# meta_data = {
#     'type': 'IQ',
#     'plot_properties':
#         {
#             'x_label': 'I',
#             'y_label': 'Q',
#             'title': 'IQ Clouds'
#         },
#     'experiment_properties':
#         {
#             'qubit': qubit,
#             'repetitions': exp_repetitions,
#         },
#     'qubit_parameters': qubit_parameters,
# }
#
# data = {
#     'I_g': traces[0].real,
#     'Q_g': traces[0].imag,
#     'I_e': traces[1].real,
#     'Q_e': traces[1].imag,
# }
#
# exp_data = ExperimentData(meta_data=meta_data, data=data)
# exp_data.save_data(file_name='IQ_clouds')

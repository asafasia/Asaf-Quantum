import matplotlib.pyplot as plt
from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
from helper.exp_helper import *
from helper.pulses import *
from qubit_parameters import qubit_parameters, update_qp
from helper.experiment_results import ExperimentData
from helper.plotter import Plotter


# %% devise setup
# qubit = "q3"
# exp = initialize_exp()
# device_setup = exp.create_device_setup()
# signal_map_default = exp.signal_map_default(qubit)
# exp_signals = exp.signals(qubit)

# %%
# session = Session(device_setup=device_setup)
# session.connect(do_emulation=False)

# %% parameters
# long_pulse = False
# simulate = False
# exp_repetitions = 1000
#
# span = 200e6
# steps = 101
#
# central_frequency = qubit_parameters[qubit]["res_freq"]
#
# dfs = np.linspace(
#     start=central_frequency - span / 2,
#     stop=central_frequency + span / 2,
#     num=steps
# )
#
# res_LO = qubit_parameters[qubit]["res_lo"]
# freq_sweep_q0 = SweepParameter(uid='freq_RF_sweep', values=dfs - res_LO)


# %% experiment
def resonator_spectroscopy(qubit, signals, exp_repetitions, freq_sweep, long_pulse=False):
    exp_qspec = Experiment(
        uid="Qubit Spectroscopy",
        signals=signals

    )
    with exp_qspec.acquire_loop_rt(
            uid="freq_shots",
            count=exp_repetitions,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
    ):
        for i in range(2):
            with exp_qspec.sweep(parameter=freq_sweep):
                with exp_qspec.section():
                    if i == 1:
                        if long_pulse:
                            exp_qspec.play(signal=f"drive_{qubit}", pulse=spec_pulse(qubit))
                        else:
                            exp_qspec.play(signal=f"drive_{qubit}", pulse=pi_pulse(qubit))



                    else:
                        exp_qspec.delay(signal=f"drive_{qubit}", time=0e-9)
                with exp_qspec.section():
                    exp_qspec.reserve(f"drive_{qubit}")
                    exp_qspec.play(signal="measure",
                                   pulse=readout_pulse(qubit),
                                   phase=qubit_parameters[qubit]['angle'])

                    exp_qspec.acquire(
                        signal="acquire",
                        handle=f"spec_{i + 1}",
                        length=qubit_parameters[qubit]["res_len"],
                    )
                with exp_qspec.section():
                    exp_qspec.delay(signal="measure", time=120e-6)
    return exp_qspec

# %% calibration
# exp_calibration = Calibration()
#
# exp_calibration["measure"] = SignalCalibration(
#     oscillator=Oscillator(
#         "readout_osc",
#         frequency=freq_sweep_q0,
#         modulation_type=ModulationType.HARDWARE,
#     ),
# )
#
# exp_qspec = resonator_spectroscopy(freq_sweep_q0)
# exp_qspec.set_calibration(exp_calibration)
# signal_map_default = {
#     f"drive_{qubit}": device_setup.logical_signal_groups[qubit].logical_signals["drive_line"],
#     "measure": device_setup.logical_signal_groups[qubit].logical_signals["measure_line"],
#     "acquire": device_setup.logical_signal_groups[qubit].logical_signals["acquire_line"],
# }
# exp_qspec.set_signal_map(signal_map_default)
#
# # # %% compile
#
# compiled_qspec = session.compile(exp_qspec)
# if simulate:
#     plot_simulation(compiled_qspec, )
# # %% run
# res_spec = session.run(compiled_qspec)
#
# # %% plot
# acquire_results_1 = res_spec.get_data("spec_1")
#
# acquire_results_2 = res_spec.get_data("spec_2")
#
# amplitude_1 = np.abs(acquire_results_1)
# amplitude_2 = np.abs(acquire_results_2)
#
# phase_radians_1 = np.unwrap(np.angle(acquire_results_1)) * amplitude_1
# phase_radians_2 = np.unwrap(np.angle(acquire_results_2)) * amplitude_2
#
# # Convert res_freq to GHz
#
# # Create a figure with 2 subplots stacked vertically
# fig, axes = plt.subplots(3, 1, figsize=(8, 6))
# drive_amp = qubit_parameters[qubit]['drive_amp']
# res_amp = qubit_parameters[qubit]['res_amp']
# res_len = qubit_parameters[qubit]['res_len']
#
# fig.suptitle(
#     f'Resonator Spectroscopy {qubit} \n drive amp = {drive_amp} V \n res amp = {res_amp:.2f} V \n res len = {res_len * 1e6} us',
#     fontsize=18)
#
# # Plot the amplitude in the first subplot
# axes[0].plot(dfs * 1e-9, amplitude_1, color='blue', marker='.', label='with_drive')
# axes[0].plot(dfs * 1e-9, amplitude_2, color='green', marker='.', label='without_drive')
#
# axes[0].set_xlabel('Frequency [GHz]')
# axes[0].set_ylabel('Amplitude [a.u.]')
# axes[0].grid(True)
#
# # Plot the phase in radians in the second subplot
# axes[1].plot(dfs * 1e-9, phase_radians_1, color='red', marker='.', label='with_drive')
# axes[1].plot(dfs * 1e-9, phase_radians_2, color='yellow', marker='.', label='without_drive')
# axes[1].set_xlabel('Frequency [GHz]')
# axes[1].set_ylabel('Phase [rad]')
# axes[1].grid(True)
# axes[0].axvline(x=qubit_parameters[qubit]["res_freq"] * 1e-9, color='green', linestyle='--')
# axes[1].axvline(x=qubit_parameters[qubit]["res_freq"] * 1e-9, color='green', linestyle='--')
#
# axes[1].legend()
# axes[0].legend()
#
# axes[2].plot(dfs * 1e-9, np.abs(amplitude_1 - amplitude_2), color='blue', marker='.', label='with_drive')
# axes[2].axvline(x=qubit_parameters[qubit]["res_freq"] * 1e-9, color='green', linestyle='--')
#
# plt.tight_layout()
# plt.show()
# # %%
#
#
# diff = np.abs(amplitude_1 - amplitude_2)
# df_var = np.sqrt(np.var(diff))
# max_index = np.argmax(diff)
#
# max_f = dfs[max_index]
#
# print(f"res freq = {dfs[max_index] * 1e-9} GHz")
# if max(diff) / df_var > 4:
#     print('updated !!!!!!!!!!')
#     update_qp(qubit, 'res_freq', max_f)
# else:
#     print('not updated')
#
# # %% save
#
# meta_data = {
#     'type': 'Resonator Spectroscopy',
#
#     'plot_properties':
#         {
#             'x_label': 'Resonator Frequency [GHz]',
#             'y_label': 'Amplitude [a.u.]',
#             'title': 'Resonator Spectroscopy'
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
#     'x': dfs,
#     'a_e': amplitude_1,
#     'a_g': amplitude_2,
#     'p_e': phase_radians_1,
#     'p_g': phase_radians_2,
# }
#
# exp_data = ExperimentData(meta_data=meta_data, data=data)
# exp_data.save_data(file_name='Resonator_Spectroscopy')
#
# # exp_plot = Plotter(meta_data=meta_data, data=data)
# # exp_plot.plot()
# # plt.show()

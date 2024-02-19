from laboneq.simple import *
from helper import pulses
from qubit_parameters import qubit_parameters


# %% Experiment Definition
def power_rabi_experiment(qubit, amplitude_sweep, exp_signals, exp_repetitions, pis_number):
    power_rabi = Experiment(
        uid="Amplitude Rabi",
        signals=exp_signals
    )

    with power_rabi.acquire_loop_rt(
            uid="rabi_shots",
            count=exp_repetitions,
            acquisition_type=AcquisitionType.SPECTROSCOPY,

    ):
        with power_rabi.sweep(uid="rabi_sweep", parameter=amplitude_sweep):
            with power_rabi.section(
                    uid="qubit_excitation"
            ):
                power_rabi.play(signal=f"drive_{qubit}",
                                pulse=pulses.many_pi_pulse(qubit, pis_number),
                                amplitude=amplitude_sweep,

                                )

            with power_rabi.section(uid="readout_section", play_after="qubit_excitation"):
                power_rabi.play(signal="measure",
                                pulse=pulses.readout_pulse(qubit),
                                phase=qubit_parameters[qubit]['angle'])

                power_rabi.acquire(
                    signal="acquire",
                    handle="amp_rabi",
                    length=qubit_parameters[qubit]["res_len"],
                )
            with power_rabi.section(uid="delay"):
                power_rabi.delay(signal="measure", time=1e-6)
    return power_rabi

# meta_data = {
#     'type': '1d',
#     'plot_properties':
#         {
#             'x_label': 'Rabi Amplitude [s]',
#             'y_label': 'State',
#             'title': 'Power Rabi Experiment'
#         },
#     'experiment_properties':
#         {
#             'qubit': qubit,
#             'repetitions': exp_repetitions,
#         },
#     'qubit_parameters': qubit_parameters[qubit],
# }
#
# data = {
#     'x_data': rabi_amp,
#     'y_data': amplitude
# }
#
# exp_data = ExperimentData(meta_data=meta_data, data=data)
# exp_data.save_data(file_name='power_rabi')

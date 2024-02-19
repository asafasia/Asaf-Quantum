# LabOne Q:
from laboneq.simple import *

# Helpers:
from laboneq.contrib.example_helpers.plotting.plot_helpers import (
    plot_result_2d,
    plot_simulation,
)
from matplotlib import pyplot as plt

descriptor = """\
instruments:
  HDAWG:
  - address: DEV8001
    uid: device_hdawg
  UHFQA:
  - address: DEV2001
    uid: device_uhfqa
  PQSC:
  - address: DEV10001
    uid: device_pqsc
connections:
  device_hdawg:
    - iq_signal: q0/drive_line
      ports: [SIGOUTS/0, SIGOUTS/1]
    - iq_signal: q1/drive_line
      ports: [SIGOUTS/2, SIGOUTS/3]
    - rf_signal: q0/flux_line
      ports: [SIGOUTS/4]
    - rf_signal: q1/flux_line
      ports: [SIGOUTS/5]
    - to: device_uhfqa
      port: DIOS/0
  device_uhfqa:
    - iq_signal: q0/measure_line
      ports: [SIGOUTS/0, SIGOUTS/1]
    - acquire_signal: q0/acquire_line
    - iq_signal: q1/measure_line
      ports: [SIGOUTS/0, SIGOUTS/1]
    - acquire_signal: q1/acquire_line
  device_pqsc:
    - to: device_hdawg
      port: ZSYNCS/0
"""

# functions that modifies the calibration on a given device setup


def calibrate_devices(device_setup):
    ## qubit 0
    # calibration setting for drive line for qubit 0
    device_setup.logical_signal_groups["q0"].logical_signals[
        "drive_line"
    ].calibration = SignalCalibration(
        # oscillator settings - frequency and type of oscillator used to modulate the pulses applied through this signal line
        oscillator=Oscillator(
            uid="drive_q0_osc", frequency=1e8, modulation_type=ModulationType.HARDWARE
        ),
        # mixer calibration settings to compensate for non-ideal mixer configuration
        mixer_calibration=MixerCalibration(
            voltage_offsets=[0.0, 0.0],
            correction_matrix=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        ),
        # global and static delay of logical signal line: use to align pulses and compensate skew
        port_delay=0,  # applied to corresponding instrument node, bound to hardware limits
        delay_signal=0,  # inserted in sequencer code, bound to waveform granularity
    )
    # calibration setting for flux line for qubit 0
    device_setup.logical_signal_groups["q0"].logical_signals[
        "flux_line"
    ].calibration = SignalCalibration(
        oscillator=Oscillator(
            uid="flux_q0_osc", frequency=1e8, modulation_type=ModulationType.HARDWARE
        ),
        # global and static delay of logical signal line: use to align pulses and compensate skew
        port_delay=0,  # applied to corresponding instrument node, bound to hardware limits
        delay_signal=0,  # inserted in sequencer code, bound to waveform granularity
    )
    # calibration setting for readout pulse line for qubit 0
    device_setup.logical_signal_groups["q0"].logical_signals[
        "measure_line"
    ].calibration = SignalCalibration(
        oscillator=Oscillator(
            uid="measure_q0_osc", frequency=1e8, modulation_type=ModulationType.SOFTWARE
        ),
        delay_signal=0,  # inserted in sequencer code, bound to waveform granularity
    )
    # calibration setting for data acquisition line for qubit 0
    device_setup.logical_signal_groups["q0"].logical_signals[
        "acquire_line"
    ].calibration = SignalCalibration(
        oscillator=Oscillator(
            uid="acquire_osc", frequency=1e8, modulation_type=ModulationType.SOFTWARE
        ),
        # delays the start of integration in relation to the start of the readout pulse to compensate for signal propagation time
        port_delay=10e-9,  # applied to corresponding instrument node, bound to hardware limits
        delay_signal=0,  # inserted in sequencer code, bound to waveform granularity
    )
    ## qubit 1
    # calibration setting for drive line for qubit 1
    device_setup.logical_signal_groups["q1"].logical_signals[
        "drive_line"
    ].calibration = SignalCalibration(
        oscillator=Oscillator(
            uid="drive_q1_osc", frequency=0.5e8, modulation_type=ModulationType.HARDWARE
        ),
        mixer_calibration=MixerCalibration(
            voltage_offsets=[0.0, 0.0],
            correction_matrix=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        ),
        # global and static delay of logical signal line: use to align pulses and compensate skew
        port_delay=0,
        delay_signal=0,
    )
    # calibration setting for flux line for qubit 1
    device_setup.logical_signal_groups["q1"].logical_signals[
        "flux_line"
    ].calibration = SignalCalibration(
        oscillator=Oscillator(
            uid="flux_q1_osc", frequency=0.5e8, modulation_type=ModulationType.HARDWARE
        ),
        # global and static delay of logical signal line: use to align pulses and compensate skew
        port_delay=0,
        delay_signal=0,
    )
    # calibration setting for readout pulse line for qubit 0
    device_setup.logical_signal_groups["q1"].logical_signals[
        "measure_line"
    ].calibration = SignalCalibration(
        oscillator=Oscillator(
            uid="measure_q1_osc",
            frequency=0.5e8,
            modulation_type=ModulationType.SOFTWARE,
        ),
        delay_signal=0,
    )
    # calibration setting for data acquisition line for qubit 0
    device_setup.logical_signal_groups["q1"].logical_signals[
        "acquire_line"
    ].calibration = SignalCalibration(
        oscillator=Oscillator(
            uid="acquire_q1_osc",
            frequency=0.5e8,
            modulation_type=ModulationType.SOFTWARE,
        ),
        # delays the start of integration in relation to the start of the readout pulse to compensate for signal propagation time
        port_delay=10e-9,
        delay_signal=0,
    )


# Function returning a calibrated device setup


def create_device_setup():
    device_setup = DeviceSetup.from_descriptor(
        descriptor,
        server_host="my_ip_address",  # ip address of the LabOne dataserver used to communicate with the instruments
        server_port="8004",  # port number of the dataserver - default is 8004
        setup_name="my_QCCS_setup",  # setup name
    )
    calibrate_devices(device_setup)
    return device_setup


# create device setup
device_setup = create_device_setup()
# use emulation mode - change, if running on hardware
use_emulation = True

## define pulses
n_qubits = 2

# qubit drive pulse - unit amplitude, but will be scaled with sweep parameter - here use the same pulse for both qubits, can be different
x90 = pulse_library.gaussian(uid="x90", length=100e-9, amplitude=1.0)
# readout drive pulse
readout_pulse = pulse_library.const(
    uid="readout_pulse", length=400e-9, amplitude=1.0 / n_qubits
)
# readout integration weights
readout_weighting_function = pulse_library.const(
    uid="readout_weighting_function", length=400e-9, amplitude=1.0
)

## define calibration settings for readout and drive - set here into the baseline calibration on DeviceSetup
lsg = device_setup.logical_signal_groups["q0"].logical_signals
lsg["drive_line"].calibration.oscillator.frequency = 100e6
lsg["drive_line"].oscillator.modulation_type = ModulationType.HARDWARE
lsg["measure_line"].calibration.oscillator.frequency = 100e6
lsg["measure_line"].oscillator.modulation_type = ModulationType.SOFTWARE
lsg["acquire_line"].calibration.port_delay = 20e-9
lsg["acquire_line"].calibration.oscillator.frequency = 100e6
lsg["acquire_line"].oscillator.modulation_type = ModulationType.SOFTWARE

lsg = device_setup.logical_signal_groups["q1"].logical_signals
lsg["drive_line"].calibration.oscillator.frequency = 50e6
lsg["drive_line"].oscillator.modulation_type = ModulationType.HARDWARE
lsg["measure_line"].calibration.oscillator.frequency = 50e6
lsg["measure_line"].oscillator.modulation_type = ModulationType.SOFTWARE
lsg["acquire_line"].calibration.port_delay = 20e-9
lsg["acquire_line"].calibration.oscillator.frequency = 50e6
lsg["acquire_line"].oscillator.modulation_type = ModulationType.SOFTWARE


# define the pulse sequence of a Rabi experiment


def rabi_pulses(
    exp,
    drive_id,
    measure_id,
    acquire_id,
    acquire_handle,
    sweep_parameter,
    excitation_pulse=x90,
    measure_pulse=readout_pulse,
    readout_weights=readout_weighting_function,
):
    # qubit excitation - pulse amplitude will be swept
    with exp.section():
        exp.play(signal=drive_id, pulse=excitation_pulse, amplitude=sweep_parameter)
    # qubit readout pulse and data acquisition
    with exp.section():
        exp.reserve(signal=drive_id)
        # play readout pulse
        exp.play(signal=measure_id, pulse=measure_pulse)
        # signal data acquisition
        exp.acquire(
            signal=acquire_id,
            handle=acquire_handle,
            kernel=readout_weights,
        )
    # relax time after readout - for signal processing and qubit relaxation to groundstate
    with exp.section():
        exp.delay(signal=measure_id, time=1e-6)

# set up sweep parameter - drive amplitude - different for the two qubits, but needs same length
count = 10
# qubit 0
start = 0
stop = 0
sweep_parameter_q0 = LinearSweepParameter(
    uid="amplitude_q0", start=start, stop=stop, count=count
)
# qubit 1
start = 0
stop = 1
sweep_parameter_q1 = LinearSweepParameter(
    uid="amplitude_q1", start=start, stop=stop, count=count
)

# number of averages
average_exponent = 10  # used for 2^n averages, n=average_exponent, maximum: n = 17

# Create Experiment
exp = Experiment(
    uid="Amplitude Rabi for two",
    signals=[
        ExperimentSignal("drive_q0"),
        ExperimentSignal("measure_q0"),
        ExperimentSignal("acquire_q0"),
        ExperimentSignal("drive_q1"),
        ExperimentSignal("measure_q1"),
        ExperimentSignal("acquire_q1"),
    ],
)
## experimental pulse sequence
# outer loop - real-time, cyclic averaging in standard integration mode
with exp.acquire_loop_rt(
    uid="shots",
    count=pow(2, average_exponent),
    averaging_mode=AveragingMode.CYCLIC,
    acquisition_type=AcquisitionType.INTEGRATION,
):
    # inner loop - real-time sweep of qubit drive pulse amplitude
    with exp.sweep(uid="sweep", parameter=[sweep_parameter_q0, sweep_parameter_q1]):
        # rabi for qubit 0
        rabi_pulses(
            exp,
            drive_id="drive_q0",
            measure_id="measure_q0",
            acquire_id="acquire_q0",
            acquire_handle="q0",
            sweep_parameter=sweep_parameter_q0,
        )
        # rabi for qubit 1
        rabi_pulses(
            exp,
            drive_id="drive_q0",
            measure_id="measure_q1",
            acquire_id="acquire_q1",
            acquire_handle="q1",
            sweep_parameter=sweep_parameter_q1,
        )

# define signal maps for qubits 0 and 1
map_q0q1 = {
    "drive_q0": device_setup.logical_signal_groups["q0"].logical_signals["drive_line"],
    "measure_q0": device_setup.logical_signal_groups["q0"].logical_signals[
        "measure_line"
    ],
    "acquire_q0": device_setup.logical_signal_groups["q0"].logical_signals[
        "acquire_line"
    ],
    "drive_q1": device_setup.logical_signal_groups["q1"].logical_signals["drive_line"],
    "measure_q1": device_setup.logical_signal_groups["q1"].logical_signals[
        "measure_line"
    ],
    "acquire_q1": device_setup.logical_signal_groups["q1"].logical_signals[
        "acquire_line"
    ],
}

# set signal map to qubit 0
exp.set_signal_map(map_q0q1)

# create and connect to session
session = Session(device_setup=device_setup)
session.connect(do_emulation=use_emulation)

# run experiment on both qubit 0 and qubit 1
my_results = session.run(exp)

# Plot simulated output signals
# plot_simulation(session.compiled_experiment, start_time=0, length=10e-6)

# plot measurement results - qubit 0
plot_result_2d(my_results, "q0", mult_axis=1)

plt.show()

# plot measurement results - qubit 1
plot_result_2d(my_results, "q1", mult_axis=1)

plt.show()
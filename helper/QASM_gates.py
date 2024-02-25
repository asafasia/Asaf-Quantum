from laboneq._utils import id_generator
from math import pi
from helper import pulses
from helper.pulses import *
from qubit_parameters import qubit_parameters


def x(qubit: Qubit, label):
    return pulse_library.const(
        uid=f"{qubit.uid}_{label}",
        length=qubit.parameters.user_defined["pulse_length"],
        amplitude=qubit.parameters.user_defined["amplitude_pi"],
    )


def sx(qubit: Qubit, label):
    return pulse_library.const(
        uid=f"{qubit.uid}_{label}",
        length=qubit.parameters.user_defined["pulse_length"],
        amplitude=qubit.parameters.user_defined["amplitude_pi"] / 2
    )


def y(qubit: Qubit, label):
    return pulse_library.const(
        uid=f"{qubit.uid}_{label}",
        length=qubit.parameters.user_defined["pulse_length"],
        amplitude=qubit.parameters.user_defined["amplitude_pi"],
        phase=pi
    )


def rz(qubit: Qubit):
    def rz_gate(angle: float):
        gate = Section(uid=id_generator(f"p_{qubit.uid}_rz_{int(180 * angle / pi)}"))
        gate.play(
            signal=qubit.signals["drive"],
            pulse=None,
            increment_oscillator_phase=angle,
        )
        return gate

    return rz_gate


def measurement(qubit: Qubit, kernel):
    def measurement_gate(handle: str):
        gate = Section(uid=id_generator(f"meas_{qubit.uid}_{handle}"),
                       trigger={"measure": {"state": True}})

        gate.reserve(signal=qubit.signals["drive"])

        gate.play(signal=qubit.signals["measure"], pulse=readout_pulse(qubit.uid),
                  phase=qubit.parameters.user_defined["angle"])

        gate.acquire(
            signal=qubit.signals["acquire"],
            handle=handle,
            kernel=kernel,
        )
        gate.delay(signal=qubit.signals["measure"], time=120e-6)

        return gate

    return measurement_gate


def cz(control: Qubit, target: Qubit, coupler: Qubit):
    def cz_gate():
        cz_id = f"cz_{control.uid}_{target.uid}"
        c_object = coupler
        gate = Section(uid=id_generator(cz_id))

        flux_section = Section(uid=id_generator('flux_pulse'))

        flux_section.play(
            signal=c_object.signals["flux"],
            pulse=flux_pulse(c_object.uid),
            length=80e-9
        )

        gate.add(flux_section)

        phase_shift_cancel_section = Section(uid=id_generator(f"p_{control.uid}_phase_shift_cancellation"),
                                             play_after=flux_section)

        phase_shift_cancel_section.play(
            signal=control.signals["drive"],
            pulse=None,
            increment_oscillator_phase=qubit_parameters[control.uid]['cz_phase_shift'],
        )

        phase_shift_cancel_section.play(
            signal=control.signals["drive"],
            pulse=pulses.pi_pulse(control.uid),
        )

        phase_shift_cancel_section.play(
            signal=target.signals["drive"],
            pulse=None,
            increment_oscillator_phase=qubit_parameters[target.uid]['cz_phase_shift'],
        )
        gate.add(phase_shift_cancel_section)

        return gate

    return cz_gate


def u_gate(qubit: Qubit, theta, phi, lam):
    def u_gate():
        gate = Section(uid=id_generator(f"u_{qubit.uid}_{int(180 * theta / pi)}"))
        gate.play(
            signal=qubit.signals["drive"],
            pulse=None,
            increment_oscillator_phase=theta,
        )
        return gate

    return u_gate
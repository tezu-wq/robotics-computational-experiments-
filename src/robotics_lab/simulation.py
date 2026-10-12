
"""Numerical simulation of a controlled rotational system."""

from dataclasses import dataclass

import numpy as np

from .dynamics import RotationalSystem


@dataclass
class SimulationResult:
    time: np.ndarray
    target: np.ndarray
    angle: np.ndarray
    angular_velocity: np.ndarray
    torque: np.ndarray


def simulate(system: RotationalSystem, controller, target_angle: float, duration: float,
             dt: float, initial_angle: float = 0.0, initial_velocity: float = 0.0,
             disturbance_torque: float = 0.0) -> SimulationResult:
    """Run forward Euler integration and return all sampled state and input values."""
    if duration <= 0 or dt <= 0:
        raise ValueError("duration and dt must be positive")
    steps = int(np.ceil(duration / dt))
    time = np.arange(steps + 1, dtype=float) * dt
    time[-1] = duration
    actual_dt = np.diff(time)
    angle = np.empty(steps + 1)
    velocity = np.empty(steps + 1)
    torque = np.empty(steps + 1)
    angle[0], velocity[0] = initial_angle, initial_velocity
    if hasattr(controller, "reset"):
        controller.reset()
    for i, step_dt in enumerate(actual_dt):
        torque[i] = controller(target_angle, angle[i], velocity[i], step_dt)
        acceleration = system.acceleration(velocity[i], torque[i] + disturbance_torque)
        velocity[i + 1] = velocity[i] + acceleration * step_dt
        angle[i + 1] = angle[i] + velocity[i] * step_dt
    torque[-1] = controller(target_angle, angle[-1], velocity[-1], dt)
    return SimulationResult(time, np.full_like(time, target_angle), angle, velocity, torque)

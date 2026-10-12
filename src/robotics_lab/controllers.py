"""Feedback controllers for the rotational system."""

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_continuous_are


@dataclass
class PIDController:
    """PID controller with optional integral and output limits."""

    kp: float = 8.0
    ki: float = 0.0
    kd: float = 3.0
    output_limit: float | None = 20.0
    integral_limit: float | None = 10.0
    _integral: float = 0.0

    def reset(self) -> None:
        self._integral = 0.0

    def __call__(self, target: float, angle: float, angular_velocity: float, dt: float) -> float:
        if dt <= 0:
            raise ValueError("dt must be positive")
        error = target - angle
        self._integral += error * dt
        if self.integral_limit is not None:
            limit = abs(self.integral_limit)
            self._integral = max(-limit, min(limit, self._integral))
        output = self.kp * error + self.ki * self._integral - self.kd * angular_velocity
        if self.output_limit is not None:
            limit = abs(self.output_limit)
            output = max(-limit, min(limit, output))
        return output


@dataclass
class LQRController:
    """Continuous-time LQR state feedback for the rotational system."""

    inertia: float = 1.0
    damping: float = 0.1
    q_angle: float = 64.0
    q_velocity: float = 4.0
    r_weight: float = 1.0
    output_limit: float | None = 20.0
    _gain: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.inertia <= 0:
            raise ValueError("inertia must be positive")
        if self.damping < 0:
            raise ValueError("damping must be non-negative")
        if self.q_angle <= 0 or self.q_velocity < 0 or self.r_weight <= 0:
            raise ValueError("LQR weights require q_angle > 0, q_velocity >= 0, and r_weight > 0")

        # x = [angle error, angular velocity], and x_dot = A x + B torque.
        a = np.array([[0.0, 1.0], [0.0, -self.damping / self.inertia]])
        b = np.array([[0.0], [1.0 / self.inertia]])
        q = np.diag([self.q_angle, self.q_velocity])
        r = np.array([[self.r_weight]])
        riccati = solve_continuous_are(a, b, q, r)
        self._gain = np.linalg.solve(r, b.T @ riccati)

    def __call__(self, target: float, angle: float, angular_velocity: float, dt: float) -> float:
        """Return saturated LQR torque; dt is accepted for controller compatibility."""
        del dt
        # -K[angle - target, angular_velocity] = K[target - angle, -angular_velocity].
        state = np.array([target - angle, -angular_velocity])
        output = float((self._gain @ state).item())
        if self.output_limit is not None:
            limit = abs(self.output_limit)
            output = max(-limit, min(limit, output))
        return output

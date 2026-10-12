
"""Performance metrics for tracking experiments."""

import numpy as np

from .simulation import SimulationResult


def evaluate(result: SimulationResult, tolerance: float = 0.02) -> dict[str, float | bool]:
    """Compute tracking error, settling time, control effort, peak torque, and success."""
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    error = result.target - result.angle
    absolute = np.abs(error)
    settled = absolute <= tolerance
    settling_time = float("nan")
    for i in range(len(settled)):
        if np.all(settled[i:]):
            settling_time = float(result.time[i])
            break
    return {
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "final_error": float(error[-1]),
        "settling_time": settling_time,
        "control_effort": float(np.trapezoid(result.torque ** 2, result.time)),
        "peak_abs_torque": float(np.max(np.abs(result.torque))),
        "success": bool(settled[-1]),
    }

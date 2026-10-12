
"""Simple rotational dynamics model."""

from dataclasses import dataclass


@dataclass
class RotationalSystem:
    """A one-degree-of-freedom rotational system."""

    inertia: float = 1.0
    damping: float = 0.1

    def __post_init__(self) -> None:
        if self.inertia <= 0:
            raise ValueError("inertia must be positive")
        if self.damping < 0:
            raise ValueError("damping must be non-negative")

    def acceleration(
        self,
        angular_velocity: float,
        torque: float,
    ) -> float:
        """Calculate angular acceleration."""
        return (
            torque - self.damping * angular_velocity
        ) / self.inertia



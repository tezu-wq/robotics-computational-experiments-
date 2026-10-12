# Robotics Computational Experiments

## Overview and research question

This repository provides reproducible computational experiments for robotics and control. Its current study asks: **How do PD, PID, and LQR tracking performance and control effort change under rotational inertia error and external torque disturbance?**

The plant is a one-degree-of-freedom rotational system. The experiment sweeps inertia ratios and disturbance limits, then compares controllers using repeated, paired trials. See [the project structure and experiment notes](docs/PROJECT_STRUCTURE.md) for the model assumptions and interpretation guidance.

## Tools
Python, NumPy, SciPy, Git

## Robustness comparison experiment

The current study compares PD, PID, and LQR control for a one-degree-of-freedom rotational model. It sweeps the true-to-nominal inertia ratio and the disturbance-torque limit, then runs repeated trials at each condition. The same sampled disturbance is applied to all controllers in a repetition (paired comparison). For each disturbance limit `L`, a trial uses one constant torque sampled uniformly from `[-L, +L]`.

Run the study from the repository root after installing the project dependencies:

```powershell
python scripts/run_robustness.py
```

The fixed random seed in `experiments/robustness.yaml` makes the generated comparison reproducible. Edit that file to change the inertia ratios, disturbance levels, repetition count, simulation settings, or controller weights. The run writes:

- `results/robustness_trials.csv`: one row per controller and repetition, including the sampled disturbance and individual metrics.
- `results/robustness_summary.csv`: mean and sample standard deviation for each controller and condition. Settling-time statistics use only repetitions that settled; `settling_time_n` records how many.
- `results/robustness_curves.png`: metrics across inertia ratios, with separate curves for each disturbance level and shaded ±1 standard deviation.
- `results/robustness_tradeoff.png`: mean RMSE versus mean control effort across conditions.

The study reports MAE, RMSE, signed final error, settling time, integrated squared control effort, peak absolute torque, and success rate. Lower error and input effort are generally desirable, but a lower value in one metric can come at the cost of another.

## Limitations
This is an idealized single-axis rigid-body model with constant inertia and viscous damping. It assumes perfect angle and velocity measurements, no sensor noise or delay, no backlash or Coulomb friction, and no actuator dynamics; only a symmetric torque-command limit is modeled. The inertia is varied while damping stays nominal. The disturbance is a constant torque per trial, randomly sampled from a bounded uniform distribution; it is not a time-varying wind or vibration model. The controller targets a fixed step angle and the simulation uses forward Euler integration. These results compare behavior within those assumptions and do not directly predict performance on a physical robot.

## References
References and technical resources will be documented as the project develops.

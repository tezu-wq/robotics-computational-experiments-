
"""Load experiment configuration and persist simulation results."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .controllers import LQRController, PIDController
from .dynamics import RotationalSystem
from .metrics import evaluate
from .simulation import simulate


ROBUSTNESS_METRICS = (
    "mae",
    "rmse",
    "final_error",
    "settling_time",
    "control_effort",
    "peak_abs_torque",
    "success",
)


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def run_experiment(config: dict, output_dir: str | Path = "results", name: str = "baseline") -> list[dict]:
    """Run configured trials and save per-sample CSV, metrics CSV, and plot."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trials = config.get("trials")
    if trials is None:
        trials = [config]
    rows = []
    summary = []
    for trial_index, trial in enumerate(trials):
        system = RotationalSystem(float(trial["inertia"]), float(trial["damping"]))
        controller = PIDController(**trial.get("controller", {}))
        result = simulate(system, controller, float(trial["target_angle"]),
                          float(trial["duration"]), float(trial["dt"]),
                          float(trial.get("initial_angle", 0.0)),
                          float(trial.get("initial_velocity", 0.0)),
                          float(trial.get("disturbance_torque", 0.0)))
        metrics = evaluate(result, float(trial.get("tolerance", 0.02)))
        label = str(trial.get("label", trial_index))
        summary.append({"trial": label, **metrics})
        rows.extend({"trial": label, "time": t, "target": target, "angle": angle,
                     "angular_velocity": velocity, "torque": torque}
                    for t, target, angle, velocity, torque in zip(result.time, result.target,
                                                                   result.angle, result.angular_velocity,
                                                                   result.torque))
        fig, ax = plt.subplots()
        ax.plot(result.time, result.target, "--", label="target")
        ax.plot(result.time, result.angle, label="angle")
        ax.set(xlabel="Time (s)", ylabel="Angle (rad)", title=f"{name}: {label}")
        ax.legend(); ax.grid(True); fig.tight_layout()
        fig.savefig(output / f"{name}_{trial_index}.png", dpi=150)
        plt.close(fig)
    with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    with (output / f"{name}_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader(); writer.writerows(summary)
    return summary


def _make_controller(spec: dict, nominal: dict):
    controller_type = str(spec["type"]).lower()
    params = dict(spec.get("parameters", {}))
    if controller_type == "pid":
        return PIDController(**params)
    if controller_type == "lqr":
        return LQRController(
            inertia=float(nominal["inertia"]),
            damping=float(nominal["damping"]),
            **params,
        )
    raise ValueError(f"Unsupported controller type: {controller_type}")


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_trials(trial_rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in trial_rows:
        key = (row["controller"], row["inertia_ratio"], row["disturbance_level"])
        groups[key].append(row)

    summary = []
    for (controller, inertia_ratio, disturbance_level), rows in sorted(groups.items()):
        aggregate = {
            "controller": controller,
            "controller_type": rows[0]["controller_type"],
            "controller_parameters": rows[0]["controller_parameters"],
            "seed": rows[0]["seed"],
            "nominal_inertia": rows[0]["nominal_inertia"],
            "damping": rows[0]["damping"],
            "target_angle": rows[0]["target_angle"],
            "duration": rows[0]["duration"],
            "dt": rows[0]["dt"],
            "tolerance": rows[0]["tolerance"],
            "inertia_ratio": inertia_ratio,
            "disturbance_level": disturbance_level,
            "repetitions": len(rows),
        }
        for metric in ROBUSTNESS_METRICS:
            values = []
            for row in rows:
                value = row[metric]
                if metric == "success":
                    values.append(float(value))
                elif value == value:  # Exclude NaN settling times for trials that never settle.
                    values.append(float(value))
            label = "success_rate" if metric == "success" else metric
            aggregate[f"{label}_mean"] = mean(values) if values else float("nan")
            aggregate[f"{label}_std"] = stdev(values) if len(values) > 1 else 0.0
            if metric == "settling_time":
                aggregate["settling_time_n"] = len(values)
        summary.append(aggregate)
    return summary


def _plot_robustness_curves(
    summary: list[dict], controllers: list[dict], inertia_ratios: list[float],
    disturbance_levels: list[float], output: Path, name: str,
) -> None:
    metrics = [
        ("rmse", "RMSE (rad)"),
        ("mae", "MAE (rad)"),
        ("settling_time", "Settling time (s)"),
        ("control_effort", "Control effort (torque²·s)"),
        ("success_rate", "Success rate"),
    ]
    fig, axes = plt.subplots(len(controllers), len(metrics), figsize=(18, 3.7 * len(controllers)), squeeze=False)
    colors = plt.get_cmap("viridis")
    for controller_index, controller in enumerate(controllers):
        controller_name = str(controller["name"])
        for metric_index, (metric, ylabel) in enumerate(metrics):
            ax = axes[controller_index, metric_index]
            for level_index, disturbance_level in enumerate(disturbance_levels):
                matching = sorted(
                    (row for row in summary
                     if row["controller"] == controller_name
                     and row["disturbance_level"] == disturbance_level),
                    key=lambda row: row["inertia_ratio"],
                )
                x = [row["inertia_ratio"] for row in matching]
                y = [row[f"{metric}_mean"] for row in matching]
                spread = [row[f"{metric}_std"] for row in matching]
                color = colors(level_index / max(1, len(disturbance_levels) - 1))
                ax.plot(x, y, marker="o", color=color, label=f"L={disturbance_level:g}")
                lower = [value - std for value, std in zip(y, spread)]
                upper = [value + std for value, std in zip(y, spread)]
                if metric == "settling_time":
                    lower = [max(0.0, value) if value == value else value for value in lower]
                if metric == "success_rate":
                    lower = [max(0.0, value) if value == value else value for value in lower]
                    upper = [min(1.0, value) if value == value else value for value in upper]
                    ax.set_ylim(0.0, 1.0)
                ax.fill_between(x, lower, upper, color=color, alpha=0.14)
            ax.set_xlabel("True / nominal inertia")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{controller_name}: {ylabel}")
            ax.grid(True, alpha=0.3)
            if controller_index == 0 and metric_index == 0:
                ax.legend(title="Disturbance limit", fontsize="small")
    fig.suptitle("Robustness across inertia error and constant disturbance", y=1.01)
    fig.tight_layout()
    fig.savefig(output / f"{name}_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_tradeoff(
    summary: list[dict], controllers: list[dict], inertia_ratios: list[float],
    disturbance_levels: list[float], output: Path, name: str,
) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=(9, 6))
    norm = Normalize(vmin=min(inertia_ratios), vmax=max(inertia_ratios))
    markers = ["o", "s", "^", "D", "v", "P"]
    max_level = max(disturbance_levels) or 1.0
    for index, controller in enumerate(controllers):
        controller_name = str(controller["name"])
        selected = [row for row in summary if row["controller"] == controller_name]
        sizes = [35 + 110 * row["disturbance_level"] / max_level for row in selected]
        ax.scatter(
            [row["rmse_mean"] for row in selected],
            [row["control_effort_mean"] for row in selected],
            c=[row["inertia_ratio"] for row in selected], cmap="plasma", norm=norm,
            s=sizes, marker=markers[index % len(markers)], alpha=0.78,
            edgecolors="black", linewidths=0.35,
        )
    ax.set_xlabel("Mean RMSE (rad) — lower is better")
    ax.set_ylabel("Mean control effort (torque²·s) — lower is better")
    ax.set_title("Tracking error versus control effort")
    ax.grid(True, alpha=0.3)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="plasma"), ax=ax)
    colorbar.set_label("True / nominal inertia")
    controller_handles = [
        Line2D([0], [0], marker=markers[i % len(markers)], linestyle="", color="black",
               label=str(controller["name"]))
        for i, controller in enumerate(controllers)
    ]
    size_handles = [
        ax.scatter([], [], s=35 + 110 * level / max_level, color="gray", edgecolors="black",
                   label=f"disturbance limit {level:g}")
        for level in disturbance_levels
    ]
    ax.legend(handles=controller_handles + size_handles, fontsize="small", loc="best")
    fig.tight_layout()
    fig.savefig(output / f"{name}_tradeoff.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_robustness_study(
    config: dict,
    output_dir: str | Path = "results",
    name: str | None = None,
) -> list[dict]:
    """Run a paired Monte Carlo comparison across inertia and disturbance levels."""
    nominal = config["nominal"]
    inertia_ratios = [float(value) for value in config["inertia_ratios"]]
    disturbance_levels = [float(value) for value in config["disturbance_levels"]]
    repetitions = int(config["repetitions"])
    controllers = config["controllers"]
    if repetitions < 2:
        raise ValueError("repetitions must be at least 2 to estimate variability")
    if not inertia_ratios or any(value <= 0 for value in inertia_ratios):
        raise ValueError("inertia_ratios must contain positive values")
    if not disturbance_levels or any(value < 0 for value in disturbance_levels):
        raise ValueError("disturbance_levels must contain non-negative values")
    if not controllers:
        raise ValueError("at least one controller must be configured")
    controller_names = [str(spec["name"]) for spec in controllers]
    if len(set(controller_names)) != len(controller_names):
        raise ValueError("controller names must be unique")

    target_angle = float(config["target_angle"])
    duration = float(config["duration"])
    dt = float(config["dt"])
    tolerance = float(config.get("tolerance", 0.02))
    seed = int(config.get("seed", 0))
    initial_angle = float(config.get("initial_angle", 0.0))
    initial_velocity = float(config.get("initial_velocity", 0.0))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    study_name = name or str(config.get("name", "robustness"))
    rng = np.random.default_rng(seed)
    controller_instances = {
        str(spec["name"]): _make_controller(spec, nominal) for spec in controllers
    }

    trial_rows = []
    for inertia_ratio in inertia_ratios:
        true_inertia = float(nominal["inertia"]) * inertia_ratio
        for disturbance_level in disturbance_levels:
            # Each replicate gets one constant load disturbance sampled uniformly from [-L, L].
            disturbances = rng.uniform(-disturbance_level, disturbance_level, size=repetitions)
            for repetition, disturbance in enumerate(disturbances):
                for spec in controllers:
                    controller_name = str(spec["name"])
                    plant = RotationalSystem(true_inertia, float(nominal["damping"]))
                    result = simulate(
                        plant, controller_instances[controller_name], target_angle, duration, dt,
                        initial_angle, initial_velocity, float(disturbance),
                    )
                    metrics = evaluate(result, tolerance)
                    trial_rows.append({
                        "controller": controller_name,
                        "controller_type": str(spec["type"]).lower(),
                        "controller_parameters": json.dumps(spec.get("parameters", {}), sort_keys=True),
                        "seed": seed,
                        "nominal_inertia": float(nominal["inertia"]),
                        "damping": float(nominal["damping"]),
                        "inertia_ratio": inertia_ratio,
                        "true_inertia": true_inertia,
                        "disturbance_level": disturbance_level,
                        "disturbance_torque": float(disturbance),
                        "target_angle": target_angle,
                        "duration": duration,
                        "dt": dt,
                        "tolerance": tolerance,
                        "repetition": repetition,
                        **metrics,
                    })

    summary = _aggregate_trials(trial_rows)
    _write_rows(output / f"{study_name}_trials.csv", trial_rows)
    _write_rows(output / f"{study_name}_summary.csv", summary)
    _plot_robustness_curves(summary, controllers, inertia_ratios, disturbance_levels, output, study_name)
    _plot_tradeoff(summary, controllers, inertia_ratios, disturbance_levels, output, study_name)
    return summary


"""Run the baseline tracking experiment."""

from pathlib import Path

from robotics_lab.experiments import load_config, run_experiment


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    summary = run_experiment(load_config(ROOT / "experiments" / "baseline.yaml"),
                             ROOT / "results", "baseline")
    print(summary[0])


if __name__ == "__main__":
    main()

"""Run the paired Monte Carlo robustness study."""

from pathlib import Path

from robotics_lab.experiments import load_config, run_robustness_study


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = load_config(ROOT / "experiments" / "robustness.yaml")
    summary = run_robustness_study(config, ROOT / "results")
    print(f"Completed {len(summary)} controller-condition combinations.")
    print(f"Summary: {ROOT / 'results' / 'robustness_summary.csv'}")
    print(f"Curves:  {ROOT / 'results' / 'robustness_curves.png'}")
    print(f"Tradeoff: {ROOT / 'results' / 'robustness_tradeoff.png'}")


if __name__ == "__main__":
    main()

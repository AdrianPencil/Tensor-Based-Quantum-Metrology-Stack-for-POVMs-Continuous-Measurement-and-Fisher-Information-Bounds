"""
Report writers.

Reports should be small, dependency-light artifacts that are easy to diff and
drop into documentation or lab notes.

Initial scope:
- a simple Markdown report for the Ramsey frequency pipeline result
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import json

from .pipelines import RamseyFrequencyPipelineResult

__all__ = [
    "write_ramsey_markdown_report",
    "write_json_report",
]


def write_ramsey_markdown_report(result: RamseyFrequencyPipelineResult, path: str | Path) -> None:
    """
    Write a compact Markdown summary of a Ramsey pipeline run.

    Args:
        result: RamseyFrequencyPipelineResult.
        path: output file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    crb_std = (result.crb_var_omega ** 0.5) if result.crb_var_omega >= 0.0 else float("nan")
    n = int(result.t_s.size)
    has_y = result.y is not None

    lines = [
        "# qmet report - Ramsey frequency pipeline",
        "",
        "## Summary",
        f"- samples: {n}",
        f"- sigma_y: {result.sigma_y:.6g}",
        f"- Fisher(omega): {result.fisher_omega:.6g}",
        f"- CRB var(omega): {result.crb_var_omega:.6g}",
        f"- CRB std(omega): {crb_std:.6g}",
        f"- simulated data: {'yes' if has_y else 'no'}",
        "",
        "## Notes",
        "- Model: mu(t) = V exp(-gamma_phi t) cos(omega t + phase0)",
        "- Fisher assumes Gaussian IID readout noise with known sigma_y",
        "",
    ]

    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json_report(payload: dict[str, Any], path: str | Path) -> None:
    """
    Write a JSON report payload.

    This is a generic helper used by workflows that want structured artifacts.

    Args:
        payload: JSON-serializable dict.
        path: output file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

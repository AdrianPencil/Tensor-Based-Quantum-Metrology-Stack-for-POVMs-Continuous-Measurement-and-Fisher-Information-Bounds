"""
Command line interface for qmet.

Design goals:
- Standard-library only (argparse) to keep dependencies minimal
- Lazy-import heavier modules so "qmet --help" stays fast
- Provide a small set of stable commands from day one

As the project grows, subcommands can map cleanly to workflows/flagship runs.
"""

import argparse
from pathlib import Path
from typing import Sequence

from . import __version__
from .core.config import GlobalConfig, save_config
from .core.linalg import comm, dagger, to_density_matrix
import numpy as np

__all__ = [
    "build_parser",
    "main",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qmet", description="qmet: quantum metrology stack")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_version = sub.add_parser("version", help="Print version")
    p_version.set_defaults(_handler=_cmd_version)

    p_config = sub.add_parser("config", help="Config utilities")
    sub_cfg = p_config.add_subparsers(dest="config_cmd", required=True)

    p_init = sub_cfg.add_parser("init", help="Write a default config JSON")
    p_init.add_argument("--out", type=str, default="qmet_config.json")
    p_init.add_argument("--seed", type=int, default=0)
    p_init.set_defaults(_handler=_cmd_config_init)

    p_smoke = sub.add_parser("smoke", help="Run a fast numeric sanity check")
    p_smoke.set_defaults(_handler=_cmd_smoke)

    return parser


def _cmd_version(args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def _cmd_config_init(args: argparse.Namespace) -> int:
    cfg = GlobalConfig(seed=int(args.seed))
    save_config(cfg, Path(args.out))
    print(f"Wrote config: {args.out}")
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    c = comm(sigma_x, sigma_z)
    if not np.allclose(c, -2.0j * np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128)):
        raise RuntimeError("commutator sanity check failed")

    psi = np.array([1.0, 1.0], dtype=np.complex128)
    psi = psi / np.linalg.norm(psi)
    rho = np.outer(psi, np.conjugate(psi))
    rho = to_density_matrix(rho)

    if not np.allclose(np.trace(rho), 1.0):
        raise RuntimeError("density matrix trace normalization failed")
    if not np.allclose(rho, dagger(rho)):
        raise RuntimeError("density matrix hermiticity failed")

    print("OK")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.error("No command selected.")
    return int(handler(args))

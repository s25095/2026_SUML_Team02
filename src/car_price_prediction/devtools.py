"""Development command wrappers for project-local tooling."""

from __future__ import annotations

import sys
from collections.abc import Sequence


DEFAULT_PYLINT_TARGETS = ["src/car_price_prediction"]


def pylint_args(argv: Sequence[str]) -> list[str]:
    """Return explicit Pylint arguments or the project default target."""

    return list(argv) or DEFAULT_PYLINT_TARGETS.copy()


def run_pylint() -> None:
    """Run Pylint against the project package when no target is provided."""

    from pylint.lint import Run  # pylint: disable=import-outside-toplevel

    Run(pylint_args(sys.argv[1:]))

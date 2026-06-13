"""Tests for project-local development command wrappers."""

from car_price_prediction.devtools import DEFAULT_PYLINT_TARGETS, pylint_args


def test_pylint_args_uses_project_default_without_arguments() -> None:
    """Pylint wrapper should lint the project package by default."""

    assert pylint_args([]) == DEFAULT_PYLINT_TARGETS


def test_pylint_args_preserves_explicit_arguments() -> None:
    """Explicit Pylint arguments should pass through unchanged."""

    assert pylint_args(["tests", "--disable=missing-function-docstring"]) == [
        "tests",
        "--disable=missing-function-docstring",
    ]

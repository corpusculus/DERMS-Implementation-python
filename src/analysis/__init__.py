"""DERMS MVP analysis package."""

from importlib import import_module

__all__ = [
    "plots",
    "kpis",
    "aggregator",
    "hosting_capacity",
    "run_phase5_analysis",
    "run_dashboard",
]


def __getattr__(name: str):
    if name in __all__:
        return import_module(f"src.analysis.{name}")
    raise AttributeError(f"module 'src.analysis' has no attribute {name!r}")

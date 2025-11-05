"""Workflow shortcuts for inventory fitting."""

from __future__ import annotations

from .censoring import fit_censored_inventory
from .hps import fit_hps_inventory

__all__ = ["fit_hps_inventory", "fit_censored_inventory"]

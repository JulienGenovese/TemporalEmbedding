"""Shared dataset variant types."""

from __future__ import annotations

from enum import Enum


class DatasetType(str, Enum):
    """Supported synthetic dataset variants."""

    SIMPLE_SPATIAL = "simple_spatial"
    SIMPLE_DELTA = "simple_delta"
    SIMPLE_CALENDAR = "simple_calendar"

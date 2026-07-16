"""Pickle-safe control messages for grounding worker shutdown."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundingDrainRequest:
	"""FIFO marker asking one grounding worker to flush and exit."""

	token: str


@dataclass(frozen=True)
class GroundingCorrectionsDrained:
	"""FIFO marker proving no earlier worker corrections remain to consume."""

	token: str

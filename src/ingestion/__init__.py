"""Standalone ingestion entrypoints."""

from ingestion.harvester import NvdHarvester, run_ingestion

__all__ = ["NvdHarvester", "run_ingestion"]

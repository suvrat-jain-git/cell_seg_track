"""plugins/base_plugin.py - abstract base class all experiment plugins implement."""
from __future__ import annotations
from abc import ABC, abstractmethod


class BasePlugin(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def enrich_fovs(self, fovs, input_dir):
        """Add experiment-specific metadata to FOVs after discovery, before segmentation."""
        ...

    @abstractmethod
    def compute_fov_features(self, spots, tracks, pop, fov):
        """Compute experiment-specific features for one FOV after population features."""
        ...

    @abstractmethod
    def export(self, spots, tracks, populations, config):
        """Write experiment-specific output files after core export."""
        ...

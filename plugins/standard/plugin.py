"""plugins/standard/plugin.py - generic cell culture plugin. No experiment-specific logic."""
from __future__ import annotations
from plugins.base_plugin import BasePlugin


class StandardPlugin(BasePlugin):
    def enrich_fovs(self, fovs, input_dir):
        return fovs

    def compute_fov_features(self, spots, tracks, pop, fov):
        return pop

    def export(self, spots, tracks, populations, config):
        pass

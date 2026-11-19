"""Wall-clock measurements for a single video processing run."""
from array import array
from collections import defaultdict
import platform
from time import perf_counter

import cv2
import numpy as np


class RuntimeProfile:
    def __init__(self):
        self.started = perf_counter()
        self.stages = defaultdict(lambda: array("d"))

    def record(self, stage, started):
        self.stages[stage].append(perf_counter() - started)

    def report(self, frames, fps, width, height, settings):
        elapsed = perf_counter() - self.started
        duration = frames / fps
        stages = {}
        for name, values in self.stages.items():
            seconds = np.frombuffer(values, dtype=np.float64)
            stages[name] = dict(calls=len(values), total_seconds=float(seconds.sum()),
                                mean_ms=float(seconds.mean()*1000),
                                p95_ms=float(np.percentile(seconds, 95)*1000))
        return dict(wall_seconds=elapsed, source_seconds=duration, frames=frames,
                    source_fps=fps, width=width, height=height, processed_fps=frames/elapsed,
                    playback_speed=duration/elapsed, stages=stages, settings=settings,
                    environment=dict(system=platform.system(), architecture=platform.machine(),
                                     python=platform.python_version(), opencv=cv2.__version__))

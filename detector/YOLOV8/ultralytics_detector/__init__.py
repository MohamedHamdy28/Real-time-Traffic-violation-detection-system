# Ultralytics YOLO 🚀, GPL-3.0 license

__version__ = "8.0.6"

from ultralytics_detector.hub import checks
from ultralytics_detector.yolo.engine.model import YOLO
from ultralytics_detector.yolo.utils import ops

__all__ = ["__version__", "YOLO", "hub", "checks"]  # allow simpler import

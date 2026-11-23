"""MiniDrive: dependency-light 2D driving environment + pluggable RL algorithms."""

from .env import MiniDriveEnv, ACTION_TABLE, ACTION_NAMES, DEFAULT_CONFIG
from .track import Track

__all__ = ["MiniDriveEnv", "ACTION_TABLE", "ACTION_NAMES", "DEFAULT_CONFIG", "Track"]
__version__ = "1.0.0"

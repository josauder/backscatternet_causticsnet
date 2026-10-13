"""Importing this package registers every model, geometry provider, weighter, task and evaluator."""

import caustics_backscatter.eval  # noqa: F401
import caustics_backscatter.geometry  # noqa: F401
import caustics_backscatter.models  # noqa: F401
import caustics_backscatter.weighting  # noqa: F401
from caustics_backscatter.tasks.backscatter import BackscatterTask
from caustics_backscatter.tasks.base import PreparedBatch, Task
from caustics_backscatter.tasks.caustics import CausticsTask

__all__ = ["BackscatterTask", "CausticsTask", "PreparedBatch", "Task"]

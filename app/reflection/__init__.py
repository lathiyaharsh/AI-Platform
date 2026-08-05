"""
Reflection Engine — evaluate and optionally improve model responses.

Public surface used by Gateway and tests.
"""

from app.reflection.models import (
    ReflectionPass,
    ReflectionResult,
    ReflectionVerdict,
)
from app.reflection.service import ReflectionService
from app.reflection.types import ReflectionDecision, ReflectionMode

__all__ = [
    "ReflectionDecision",
    "ReflectionMode",
    "ReflectionPass",
    "ReflectionResult",
    "ReflectionService",
    "ReflectionVerdict",
]

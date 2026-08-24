"""
FLARE model architectures integrated into flx.
"""
from flx.models.flare.fdd import FDD
from flx.models.flare.pose import GRIDNET4, FingerPose_2D_Single

__all__ = ["FDD", "GRIDNET4", "FingerPose_2D_Single"]

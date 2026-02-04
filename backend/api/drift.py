from fastapi import APIRouter

from backend.services.drift import DriftDetector
from backend.schemas.drift import DriftResult

router = APIRouter(prefix="/drift", tags=["drift"])

detector = DriftDetector()


@router.post("", response_model=DriftResult)
def detect_drift(
    reference: list[float],
    current: list[float],
):
    return detector.ks_drift(reference, current)

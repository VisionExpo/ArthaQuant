from pydantic import BaseModel


class DriftResult(BaseModel):
    drift_detected: bool
    p_value: float
    statistic: float

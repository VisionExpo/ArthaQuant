from pydantic import BaseModel
from datetime import datetime


class PaperTradeRequest(BaseModel):
    symbol: str
    p_up: float
    expected_return: float
    uncertainty: float
    price: float
    timestamp: datetime

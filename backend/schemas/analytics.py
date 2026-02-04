from pydantic import BaseModel
from typing import List


class PortfolioAnalytics(BaseModel):
    equity_curve: List[float]
    sharpe_ratio: float
    max_drawdown: float
    total_return: float

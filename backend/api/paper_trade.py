from fastapi import APIRouter, HTTPException

from backend.schemas.paper_trade import PaperTradeRequest
from backend.schemas.trading import TradeSignal
from backend.services.execution import PaperExecutionEngine
from backend.services.portfolio import Portfolio

router = APIRouter(prefix="/paper", tags=["paper-trading"])

# In-memory portfolio (MVP)
portfolio = Portfolio(initial_cash=1_000_000)

execution_engine = PaperExecutionEngine()


@router.post("/trade")
def paper_trade(req: PaperTradeRequest):
    signal = TradeSignal(
        symbol=req.symbol,
        p_up=req.p_up,
        expected_return=req.expected_return,
        uncertainty=req.uncertainty,
        timestamp=req.timestamp,
    )

    trade = execution_engine.generate_trade(
        signal=signal,
        current_price=req.price,
        capital=portfolio.cash,
    )

    if trade:
        portfolio.apply_trade(trade)
        return {
            "status": "executed",
            "trade": trade.dict(),
            "cash": portfolio.cash,
            "positions": portfolio.positions,
        }

    return {
        "status": "no_trade",
        "reason": "signal did not meet execution criteria",
    }

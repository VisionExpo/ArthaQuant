from fastapi import APIRouter
from backend.api.paper_trade import portfolio

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("")
def get_portfolio():
    return {
        "cash": portfolio.cash,
        "positions": portfolio.positions,
        "trades": [t.dict() for t in portfolio.trade_log],
    }

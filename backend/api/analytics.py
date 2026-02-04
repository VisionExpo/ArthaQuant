from fastapi import APIRouter

from backend.api.paper_trade import portfolio
from backend.services.analytics import (
    equity_curve_from_trades,
    sharpe_ratio,
    max_drawdown,
)
from backend.schemas.analytics import PortfolioAnalytics

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("", response_model=PortfolioAnalytics)
def get_analytics():
    if not portfolio.trade_log:
        return PortfolioAnalytics(
            equity_curve=[],
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            total_return=0.0,
        )

    # Placeholder prices (replace with real market prices later)
    prices = [trade.price for trade in portfolio.trade_log]

    equity_curve = equity_curve_from_trades(
        initial_cash=1_000_000,
        trades=portfolio.trade_log,
        price_series=prices,
    )

    return PortfolioAnalytics(
        equity_curve=equity_curve,
        sharpe_ratio=sharpe_ratio(equity_curve),
        max_drawdown=max_drawdown(equity_curve),
        total_return=(equity_curve[-1] / equity_curve[0]) - 1,
    )

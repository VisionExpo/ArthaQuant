import numpy as np


def equity_curve_from_trades(
    initial_cash: float,
    trades: list,
    price_series: list[float],
):
    equity = initial_cash
    equity_curve = []
    position = 0

    for trade, price in zip(trades, price_series):
        if trade.side == "BUY":
            equity -= trade.quantity * trade.price
            position += trade.quantity
        elif trade.side == "SELL":
            equity += trade.quantity * trade.price
            position -= trade.quantity

        equity_curve.append(equity + position * price)

    return equity_curve


def sharpe_ratio(equity_curve, risk_free_rate=0.0):
    returns = np.diff(equity_curve) / equity_curve[:-1]
    if returns.std() == 0:
        return 0.0
    return float((returns.mean() - risk_free_rate) / returns.std())


def max_drawdown(equity_curve):
    peak = equity_curve[0]
    max_dd = 0.0

    for value in equity_curve:
        peak = max(peak, value)
        dd = (peak - value) / peak
        max_dd = max(max_dd, dd)

    return float(max_dd)

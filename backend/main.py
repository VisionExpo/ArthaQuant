from fastapi import FastAPI
from backend.api import predict, paper_trade, portfolio
from backend.api import analytics
from backend.api import drift
from backend.api import auth, user_portfolio

app = FastAPI(title="ArthaQuant API")

app.include_router(predict.router)
app.include_router(paper_trade.router)
app.include_router(portfolio.router)
app.include_router(analytics.router)
app.include_router(drift.router)
app.include_router(auth.router)
app.include_router(user_portfolio.router)
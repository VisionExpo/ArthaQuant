from fastapi import FastAPI
from backend.api import predict, paper_trade, portfolio

app = FastAPI(title="ArthaQuant API")

app.include_router(predict.router)
app.include_router(paper_trade.router)
app.include_router(portfolio.router)

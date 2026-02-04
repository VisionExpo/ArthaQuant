from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.db.session import SessionLocal
from backend.db.models import Portfolio

router = APIRouter(prefix="/user/portfolio", tags=["portfolio"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("")
def create_portfolio(user_id: int, db: Session = Depends(get_db)):
    portfolio = Portfolio(owner_id=user_id)
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)
    return portfolio

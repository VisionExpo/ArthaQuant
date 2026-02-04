from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.schemas.auth import UserCreate, UserOut
from backend.services.auth import create_user
from backend.db.session import SessionLocal

router = APIRouter(prefix="/auth", tags=["auth"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/register", response_model=UserOut)
def register(user: UserCreate, db: Session = Depends(get_db)):
    return create_user(db, user.email, user.password)

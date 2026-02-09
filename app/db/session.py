import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL_ENV = "DATABASE_URL"
DEFAULT_SQLITE_URL = "sqlite:///./app.db"


class Base(DeclarativeBase):
    pass


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite:///") or url.startswith("sqlite://")


def _create_engine():
    database_url = os.getenv(DATABASE_URL_ENV, DEFAULT_SQLITE_URL)
    connect_args = {"check_same_thread": False} if _is_sqlite(database_url) else {}
    return create_engine(database_url, connect_args=connect_args)


ENGINE = _create_engine()
SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False)


@contextmanager
def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=ENGINE)

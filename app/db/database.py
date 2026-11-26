"""Database bootstrap and session management.

Uses SQLAlchemy with a MySQL connection from settings. `init_db()` is safe to call
on startup: it creates the database if missing, then creates ORM tables.
"""

import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for ORM models."""
    pass


# Main app engine; pool_pre_ping avoids stale MySQL connections after idle periods.
engine = create_engine(
    settings.mysql_url,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Ensure database exists and ORM tables are created."""
    # Connect without a database name first so we can create the target DB if needed.
    from urllib.parse import quote_plus

    user = quote_plus(settings.mysql_user)
    password = quote_plus(settings.mysql_password)
    admin_url = (
        f"mysql+pymysql://{user}:{password}"
        f"@{settings.mysql_host}:{settings.mysql_port}/?charset=utf8mb4"
    )
    admin_engine = create_engine(admin_url, pool_pre_ping=True)
    with admin_engine.connect() as conn:
        conn.execute(
            text(
                f"CREATE DATABASE IF NOT EXISTS `{settings.mysql_database}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        )
        conn.commit()
    admin_engine.dispose()

    # Import models before create_all so SQLAlchemy knows about table metadata.
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized")


@contextmanager
def get_db_session():
    """Transactional session scope for request logging and other DB writes."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

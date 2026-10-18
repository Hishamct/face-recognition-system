from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

engine = create_engine(
    settings.sync_database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def verify_db_connection() -> dict:
    """Verifies connection to PostgreSQL and returns database version."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();")).scalar()
            return {"status": "connected", "version": result}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

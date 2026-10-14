
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text
from sqlalchemy.engine import Engine

# Project root: movie-recommender/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


def get_engine() -> Engine:
    """Create a SQLAlchemy engine for PostgreSQL."""

    required = [
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ]

    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}"
        )

    database_url = URL.create(
        drivername="postgresql+psycopg",
        username=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB"),
    )

    return create_engine(database_url, pool_pre_ping=True)


def test_connection() -> None:
    """Verify that PostgreSQL is reachable."""

    engine = get_engine()

    try:
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT current_database(), version()")
            )
            database, version = result.fetchone()

            print(f"Connected to database: {database}")
            print(f"PostgreSQL version: {version}")

    finally:
        engine.dispose()


if __name__ == "__main__":
    test_connection()

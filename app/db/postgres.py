from __future__ import annotations

from urllib.parse import quote, unquote, urlparse, urlunparse

import asyncpg
from pgvector.asyncpg import register_vector

from app.config.settings import settings

_pool: asyncpg.Pool | None = None


def normalize_database_url(url: str) -> str:
    """
    Percent-encode credentials so passwords with @ * % etc. parse correctly.

    Also force sslmode=require for Supabase / managed Postgres when unset.
    """
    raw = (url or "").strip()
    if not raw:
        raise ValueError("SUPABASE_DB_URL is empty")

    if "://" not in raw:
        raise ValueError("SUPABASE_DB_URL must include a scheme (postgresql://…)")

    scheme, rest = raw.split("://", 1)
    if "@" not in rest:
        return raw

    userinfo, hostpart = rest.rsplit("@", 1)
    if ":" in userinfo:
        username, password = userinfo.split(":", 1)
    else:
        username, password = userinfo, ""

    username = quote(unquote(username), safe="")
    password = quote(unquote(password), safe="")
    encoded = f"{scheme}://{username}:{password}@{hostpart}"

    parsed = urlparse(encoded)
    query = parsed.query
    if "sslmode=" not in query.lower():
        query = f"{query}&sslmode=require" if query else "sslmode=require"
    return urlunparse(parsed._replace(query=query))


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def get_pool() -> asyncpg.Pool:
    """Shared asyncpg pool for the process."""
    global _pool
    if _pool is not None:
        return _pool

    dsn = normalize_database_url(settings.supabase_db_url)
    # asyncpg ignores libpq sslmode in some setups; force TLS for managed PG.
    needs_ssl = "supabase.com" in dsn or "sslmode=require" in dsn
    _pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=1,
        max_size=5,
        init=_init_connection,
        command_timeout=60,
        ssl="require" if needs_ssl else None,
    )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None

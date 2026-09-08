import os
from contextlib import asynccontextmanager

import asyncpg


@asynccontextmanager
async def get_connection():
    """A connection opened and closed per call.

    Deliberately not a pooled/shared connection: at this system's stated
    scale (single user, demo-sized corpus -- see ARCHITECTURE.md's
    assumptions) a pool is premature, and a shared pool object would be
    bound to whichever asyncio event loop created it, which breaks across
    pytest-asyncio's per-test event loops. Per-call connections sidestep
    that entirely.
    """
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        yield conn
    finally:
        await conn.close()

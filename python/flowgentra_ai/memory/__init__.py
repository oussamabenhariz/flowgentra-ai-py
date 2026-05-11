"""Memory and persistence layer for conversations and state.

This module provides interfaces for persisting conversation history, managing checkpoints,
and implementing memory strategies for long-running workflows.

Examples:
    Use conversation memory:

        from flowgentra_ai.memory import ConversationMemory

        memory = ConversationMemory()

    Async in-memory checkpointer (full history):

        from flowgentra_ai.memory import AsyncMemoryCheckpointer

        cp = AsyncMemoryCheckpointer()
        await cp.save("thread-1", state, meta)
        history = await cp.list_history("thread-1")

    Namespaced checkpointer:

        from flowgentra_ai.memory import AsyncMemoryCheckpointer, NamespacedCheckpointer

        inner = AsyncMemoryCheckpointer()
        cp = NamespacedCheckpointer(inner, "my_agent")

    SQLite async checkpointer:

        from flowgentra_ai.memory import AsyncSqliteCheckpointer

        cp = await AsyncSqliteCheckpointer.new("sqlite:///checkpoints.db")

    PostgreSQL async checkpointer:

        from flowgentra_ai.memory import AsyncPostgresCheckpointer

        cp = await AsyncPostgresCheckpointer.new("postgresql://user:pass@localhost/db")

    Redis async checkpointer:

        from flowgentra_ai.memory import AsyncRedisCheckpointer

        cp = await AsyncRedisCheckpointer.new("redis://localhost/", ttl_seconds=86400)

    MongoDB async checkpointer:

        from flowgentra_ai.memory import AsyncMongoCheckpointer

        cp = await AsyncMongoCheckpointer.new("mongodb://localhost:27017", "mydb", "checkpoints")

    MySQL async checkpointer:

        from flowgentra_ai.memory import AsyncMysqlCheckpointer

        cp = await AsyncMysqlCheckpointer.new("mysql://user:pass@localhost/db")
"""

from flowgentra_ai._native import memory as _m, state as _s

# ── Core types ────────────────────────────────────────────────────────────────
Checkpoint = _m.Checkpoint
CheckpointMetadata = _m.CheckpointMetadata
ConversationMemory = _m.ConversationMemory
TokenBufferMemory = _m.TokenBufferMemory
SummaryConfig = _m.SummaryConfig
SummaryMemory = _m.SummaryMemory

# ── File-based (always available) ────────────────────────────────────────────
FileCheckpointer = _s.FileCheckpointer

# ── Database-backed checkpointers ────────────────────────────────────────────
# Conditionally available based on Cargo feature flags.
# If the native extension was built without the relevant feature, returns None.

def _try_import(module, attr):
    try:
        return getattr(module, attr)
    except AttributeError:
        return None

# Sync (original) checkpointers
SqliteCheckpointer = _try_import(_s, "SqliteCheckpointer")
PostgresCheckpointer = _try_import(_s, "PostgresCheckpointer")
RedisCheckpointer = _try_import(_s, "RedisCheckpointer")

# Async in-memory checkpointer — pure Python, always available
class AsyncMemoryCheckpointer:
    """Async in-memory checkpointer with full history.

    Thread-safe for use within a single process. State is lost when the
    process exits. Intended for testing and short-lived workflows.

    Example::

        cp = AsyncMemoryCheckpointer()
        await cp.save("thread-1", {"step": 1, "output": "hello"})
        latest = await cp.load("thread-1")      # {"step": 1, "output": "hello"}
        history = await cp.list_history("thread-1")
    """

    def __init__(self) -> None:
        self._store: dict = {}

    async def save(self, thread_id: str, state: dict, metadata: dict | None = None) -> None:
        """Append a checkpoint for *thread_id*."""
        self._store.setdefault(thread_id, []).append(state)

    async def load(self, thread_id: str) -> dict | None:
        """Return the latest state for *thread_id*, or ``None``."""
        entries = self._store.get(thread_id)
        return entries[-1] if entries else None

    async def list_history(self, thread_id: str) -> list:
        """Return all saved states for *thread_id*, oldest first."""
        return list(self._store.get(thread_id, []))

    async def list_threads(self) -> list:
        """Return all thread IDs that have at least one checkpoint."""
        return list(self._store.keys())

    async def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints for *thread_id*."""
        self._store.pop(thread_id, None)

    def __repr__(self) -> str:
        return f"AsyncMemoryCheckpointer(threads={list(self._store.keys())})"

# Native override if compiled in (future-proof)
_native_async_memory = _try_import(_s, "AsyncMemoryCheckpointer")
if _native_async_memory is not None:
    AsyncMemoryCheckpointer = _native_async_memory

NamespacedCheckpointer = _try_import(_s, "NamespacedCheckpointer")
CheckpointHistoryEntry = _try_import(_s, "CheckpointHistoryEntry")

# Async DB checkpointers (feature-gated)
AsyncSqliteCheckpointer = _try_import(_s, "AsyncSqliteCheckpointer")
AsyncPostgresCheckpointer = _try_import(_s, "AsyncPostgresCheckpointer")
AsyncRedisCheckpointer = _try_import(_s, "AsyncRedisCheckpointer")
AsyncMongoCheckpointer = _try_import(_s, "AsyncMongoCheckpointer")
AsyncMysqlCheckpointer = _try_import(_s, "AsyncMysqlCheckpointer")

__all__ = [
    "Checkpoint",
    "CheckpointMetadata",
    "ConversationMemory",
    "TokenBufferMemory",
    "SummaryConfig",
    "SummaryMemory",
    # Sync checkpointers
    "FileCheckpointer",
    "SqliteCheckpointer",
    "PostgresCheckpointer",
    "RedisCheckpointer",
    # Async checkpointers
    "AsyncMemoryCheckpointer",
    "NamespacedCheckpointer",
    "CheckpointHistoryEntry",
    "AsyncSqliteCheckpointer",
    "AsyncPostgresCheckpointer",
    "AsyncRedisCheckpointer",
    "AsyncMongoCheckpointer",
    "AsyncMysqlCheckpointer",
]

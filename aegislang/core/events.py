"""
AegisLang Event Publishing Module.

Provides reliable Redis-based event publishing with retry logic
for the Agent-OS event bus integration.

Usage:
    from aegislang.core.events import publish_event

    # Publish with automatic retry
    await publish_event(
        topic="policy.ingested",
        data=document.model_dump_json(),
        redis_url="redis://localhost:6379",
    )
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Default configuration
DEFAULT_REDIS_URL = "redis://localhost:6379"
DEFAULT_MAX_RETRIES = 4
DEFAULT_BASE_DELAY = 2.0  # seconds


async def publish_event(
    topic: str,
    data: str | bytes,
    redis_url: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
) -> bool:
    """
    Publish an event to Redis with exponential backoff retry.

    Args:
        topic: The event topic/channel to publish to
        data: The event data (JSON string or bytes)
        redis_url: Redis connection URL (defaults to REDIS_URL env or localhost)
        max_retries: Maximum number of retry attempts (default 4)
        base_delay: Base delay in seconds for exponential backoff (default 2.0)

    Returns:
        True if publish succeeded, False otherwise

    Retry schedule with default base_delay=2:
        - Attempt 1: immediate
        - Attempt 2: after 2s
        - Attempt 3: after 4s
        - Attempt 4: after 8s
        - Attempt 5: after 16s (if max_retries=4)
    """
    url = redis_url or os.environ.get("REDIS_URL", DEFAULT_REDIS_URL)

    for attempt in range(max_retries + 1):
        try:
            import redis.asyncio as redis_async

            client = redis_async.from_url(url)
            try:
                await client.publish(topic, data)
                logger.debug(
                    "event_published",
                    topic=topic,
                    attempt=attempt + 1,
                    data_size=len(data) if isinstance(data, (str, bytes)) else 0,
                )
                return True
            finally:
                await client.aclose()

        except ImportError:
            logger.warning(
                "redis_not_installed",
                message="redis package not installed, event publishing disabled",
            )
            return False

        except Exception as e:
            delay = base_delay * (2 ** attempt)

            if attempt < max_retries:
                logger.warning(
                    "event_publish_retry",
                    topic=topic,
                    attempt=attempt + 1,
                    max_retries=max_retries + 1,
                    error=str(e),
                    retry_in_seconds=delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "event_publish_failed",
                    topic=topic,
                    attempts=attempt + 1,
                    error=str(e),
                    message="All retry attempts exhausted",
                )
                return False

    return False


async def publish_event_sync_wrapper(
    topic: str,
    data: str | bytes,
    redis_url: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
) -> bool:
    """
    Synchronous wrapper for publish_event.

    Use this when you need to call from synchronous code.
    Creates a new event loop if one is not running.
    """
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context, create a task
        return await publish_event(topic, data, redis_url, max_retries, base_delay)
    except RuntimeError:
        # No running loop, create one
        return asyncio.run(
            publish_event(topic, data, redis_url, max_retries, base_delay)
        )


class EventPublisher:
    """
    Reusable event publisher with connection pooling.

    For high-throughput scenarios, use this class to maintain
    a persistent connection to Redis.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
    ):
        self.redis_url = redis_url or os.environ.get("REDIS_URL", DEFAULT_REDIS_URL)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._client: Any = None

    async def connect(self) -> None:
        """Establish connection to Redis."""
        if self._client is None:
            try:
                import redis.asyncio as redis_async
                self._client = redis_async.from_url(self.redis_url)
            except ImportError:
                logger.warning("redis_not_installed")

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def publish(self, topic: str, data: str | bytes) -> bool:
        """Publish event with retry logic."""
        for attempt in range(self.max_retries + 1):
            try:
                if self._client is None:
                    await self.connect()

                if self._client is None:
                    return False

                await self._client.publish(topic, data)
                return True

            except Exception as e:
                delay = self.base_delay * (2 ** attempt)

                if attempt < self.max_retries:
                    logger.warning(
                        "event_publish_retry",
                        topic=topic,
                        attempt=attempt + 1,
                        error=str(e),
                        retry_in_seconds=delay,
                    )
                    # Reset connection on error
                    self._client = None
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "event_publish_failed",
                        topic=topic,
                        attempts=attempt + 1,
                        error=str(e),
                    )
                    return False

        return False

    async def __aenter__(self) -> "EventPublisher":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()


__all__ = [
    "publish_event",
    "publish_event_sync_wrapper",
    "EventPublisher",
    "DEFAULT_REDIS_URL",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_BASE_DELAY",
]

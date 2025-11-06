"""
Unified Subtensor Connection Manager

Manages both synchronous (thread-local) and asynchronous (singleton) subtensor connections
to ensure thread safety, connection reuse, and proper lifecycle management.

Usage:
    # Create manager
    manager = SubtensorConnectionManager(config, refresh_interval_hours=12)

    # Sync operations (thread-local)
    subtensor = manager.get_subtensor()
    metagraph = subtensor.metagraph(netuid)

    # Async operations (singleton, reused)
    async_subtensor = await manager.get_async_subtensor()
    metadata = await async_subtensor.get_metadata(netuid, hotkey)
"""

import threading
import datetime
import bittensor as bt
from typing import Optional


class SubtensorConnectionManager:
    """
    Unified manager for both sync (thread-local) and async (singleton) subtensor connections.

    Sync connections: One per thread, for metagraph queries and weight setting
    Async connection: Single reused connection for chain metadata queries
    """

    def __init__(self, config, refresh_interval_hours: int = 12):
        """
        Initialize connection manager.

        Args:
            config: Bittensor config object
            refresh_interval_hours: Hours before connection is refreshed (default: 12)
        """
        self.config = config
        self.refresh_interval = datetime.timedelta(hours=refresh_interval_hours)

        # For sync connections (thread-local)
        self.thread_local = threading.local()
        self._sync_lock = threading.Lock()

        # For async connection (singleton)
        self._async_subtensor = None
        self._async_created_at = None

        bt.logging.debug(
            f"SubtensorConnectionManager initialized with {refresh_interval_hours}h refresh interval"
        )

    # =========================================================================
    # SYNC CONNECTION METHODS (Thread-Local)
    # =========================================================================

    def get_subtensor(self) -> bt.subtensor:
        if not (hasattr(self.thread_local, 'subtensor') and hasattr(self.thread_local, 'created_at')):
            return self._create_sync_connection()

        age = datetime.datetime.now() - self.thread_local.created_at
        if age > self.refresh_interval:
            bt.logging.info(
                f"Sync subtensor connection is {age.total_seconds()/3600:.1f}h old; refreshing "
                f"(threshold: {self.refresh_interval.total_seconds()/3600:.0f}h, "
                f"thread: {threading.current_thread().name})"
            )
            return self._refresh_sync_connection()

        return self.thread_local.subtensor


    def _create_sync_connection(self) -> bt.subtensor:
        """
        Create new sync connection for current thread.

        Returns:
            bt.subtensor: Newly created sync connection

        Raises:
            Exception: If connection creation fails
        """
        try:
            bt.logging.debug(
                f"Creating sync subtensor connection for thread {threading.current_thread().name}"
            )

            # Create new connection
            subtensor = bt.subtensor(config=self.config)

            # Store in thread-local storage
            self.thread_local.subtensor = subtensor
            self.thread_local.created_at = datetime.datetime.now()

            bt.logging.success(
                f"Sync subtensor connection created for thread {threading.current_thread().name}"
            )

            return subtensor

        except Exception as e:
            bt.logging.error(
                f"Failed to create sync subtensor connection: {e}",
                exc_info=True
            )
            raise

    def _refresh_sync_connection(self) -> bt.subtensor:
        if hasattr(self.thread_local, 'subtensor'):
            try:
                self.thread_local.subtensor.close()
                bt.logging.debug("Closed old sync subtensor connection")
            except Exception as e:
                bt.logging.warning(f"Error closing old sync subtensor connection: {e}")
        return self._create_sync_connection()


    # =========================================================================
    # ASYNC CONNECTION METHODS (Singleton, Reused)
    # =========================================================================

    async def get_async_subtensor(self):
        if self._async_subtensor is None or self._async_created_at is None:
            await self._create_async_connection()
        else:
            age = datetime.datetime.now() - self._async_created_at
            if age > self.refresh_interval:
                bt.logging.info(
                    f"Async subtensor connection is {age.total_seconds()/3600:.1f}h old; refreshing "
                    f"(threshold: {self.refresh_interval.total_seconds()/3600:.0f}h)"
                )
                await self._refresh_async_connection()
        return self._async_subtensor

    async def _create_async_connection(self):
        """
        Create new async connection (singleton).

        Raises:
            Exception: If connection creation fails
        """
        try:
            bt.logging.debug("Creating async subtensor connection")

            # Create new async connection
            self._async_subtensor = bt.AsyncSubtensor(config=self.config)

            # Initialize if needed (check AsyncSubtensor API)
            if hasattr(self._async_subtensor, 'initialize'):
                await self._async_subtensor.initialize()

            self._async_created_at = datetime.datetime.now()

            bt.logging.success("Async subtensor connection created")

        except Exception as e:
            bt.logging.error(f"Failed to create async subtensor connection: {e}", exc_info=True)
            self._async_subtensor = None
            self._async_created_at = None
            raise

    async def _refresh_async_connection(self):
        if self._async_subtensor:
            try:
                await self._async_subtensor.close()
                bt.logging.debug("Closed old async subtensor connection")
            except Exception as e:
                bt.logging.warning(f"Error closing old async subtensor connection: {e}")
        await self._create_async_connection()


    # =========================================================================
    # CLEANUP METHODS
    # =========================================================================

    def close_sync_connection(self):
        """
        Close sync connection for current thread.

        Call this when a thread is shutting down to clean up its connection.
        """
        if hasattr(self.thread_local, 'subtensor'):
            try:
                if hasattr(self.thread_local.subtensor, 'close'):
                    self.thread_local.subtensor.close()
                    bt.logging.info(
                        f"Closed sync subtensor connection for thread {threading.current_thread().name}"
                    )
            except Exception as e:
                bt.logging.warning(f"Error closing sync subtensor connection: {e}")
            finally:
                # Clean up thread-local storage
                delattr(self.thread_local, 'subtensor')
                if hasattr(self.thread_local, 'created_at'):
                    delattr(self.thread_local, 'created_at')

    async def close_async_connection(self):
        """
        Close async connection.

        Must be called from async context. Call this on validator shutdown.
        """
        if self._async_subtensor:
            try:
                if hasattr(self._async_subtensor, 'close'):
                    await self._async_subtensor.close()
                    bt.logging.info("Closed async subtensor connection")
            except Exception as e:
                bt.logging.warning(f"Error closing async subtensor connection: {e}")
            finally:
                self._async_subtensor = None
                self._async_created_at = None

    def cleanup(self):
        """
        Cleanup all connections on validator shutdown.

        This closes the sync connection for the calling thread.
        Async connection should be closed separately via close_async_connection().
        """
        bt.logging.info("Cleaning up subtensor connections...")

        # Close sync connection for current thread
        self.close_sync_connection()

        # Note: Async connection cleanup requires async context
        # It should be handled separately via close_async_connection()
        # For now, just mark for cleanup
        if self._async_subtensor:
            bt.logging.warning(
                "Async subtensor connection still exists. "
                "Call close_async_connection() from async context for proper cleanup."
            )
            self._async_subtensor = None
            self._async_created_at = None

        bt.logging.info("Subtensor connection cleanup complete")

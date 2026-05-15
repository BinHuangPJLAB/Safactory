from __future__ import annotations

import asyncio
import logging
import sqlite3
from typing import Callable, Dict, List, Optional

from clusters.docker_clusters import DockerContainerBackend

from .actor_pool import DockerAgentPool
from .binding_plan import build_binding_plan
from .repository import AgentDataRepository
from .types import PoolEntry

log = logging.getLogger("manager")


class AgentPoolManager:
    """
    OpenClaw Docker scheduler facade.

    It warms Docker containers, exposes ready leases, and refills from the DB
    after each episode. No agent HTTP service is involved.
    """

    def __init__(
        self,
        cfg: dict,
        conn: Optional[sqlite3.Connection],
        *,
        job_id: str = "",
        db_processing_done_checker: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.cfg = cfg or {}
        self._job_id = str(job_id or "").strip()
        self._repo = AgentDataRepository(
            conn,
            job_id=self._job_id,
            db_processing_done_checker=db_processing_done_checker,
        )
        self._pool_size = int(self.cfg.get("pool_size", 0) or 0)
        self._mode = str(self.cfg.get("mode", "docker") or "docker").strip().lower()
        if self._mode != "docker":
            raise ValueError(f"Only docker mode is supported by the OpenClaw workflow; got {self._mode!r}")

        cluster_cfg: Dict = dict(self.cfg.get("cluster", {}) or {})
        docker_cfg: Dict = dict(cluster_cfg.get("docker", {}) or {})
        startup_concurrency = int(docker_cfg.get("startup_concurrency", 8) or 8)

        self._backend = DockerContainerBackend(cluster_cfg=cluster_cfg)
        self._pool = DockerAgentPool(
            repo=self._repo,
            docker=self._backend,
            pool_size=self._pool_size,
            startup_concurrency=startup_concurrency,
        )
        self._state_lock = asyncio.Lock()
        self._initialized = False
        self._closed = False

    async def start(self) -> None:
        async with self._state_lock:
            if self._initialized:
                return
            self._closed = False

            plan = build_binding_plan(self._repo)
            if not plan.env_to_image:
                log.warning("No agent/image mapping found in DB; nothing to start.")
                self._initialized = True
                return

            prewarm_rows = await self._repo.prime(self._pool_size) if self._pool_size > 0 else []
            await self._backend.start(plan)
            await self._pool.prewarm(rows=prewarm_rows)

            self._initialized = True
            log.info(
                "OpenClaw Docker scheduler started: pool_size=%d job_id=%s",
                self._pool_size,
                self._job_id or "<all>",
            )

    async def close_all(self) -> None:
        async with self._state_lock:
            if self._closed:
                log.info("AgentPoolManager.close_all(): already closed")
                return
            self._initialized = False
            self._closed = True
            await self._pool.reset()

        try:
            await self._backend.close()
        except Exception:
            log.warning("Docker backend close failed (ignored)", exc_info=True)

    async def list_pool_instances(self) -> List[PoolEntry]:
        return await self._pool.list_instances()

    async def close_and_refill(
        self,
        agent: str,
        id_: str,
        *,
        succeeded: bool,
    ) -> Optional[PoolEntry]:
        if not self._initialized:
            raise RuntimeError("AgentPoolManager not started. Call await start() first.")
        return await self._pool.close_and_refill(env=str(agent), env_id=str(id_), succeeded=bool(succeeded))

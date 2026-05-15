from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from core.data_manager.manager import DataManager, SessionContext

from .agent_start_client import AgentStartClient
from .simulation_lease_pool import SimulationLeasePool
from .types import (
    SimulationAgentLease,
    SimulationRunConfig,
    SimulationRunSummary,
    SimulationStartRequest,
    SimulationStartResult,
)

log = logging.getLogger("manager.simulation_worker")


class SimulationWorkerGroup:
    def __init__(
        self,
        lease_pool: SimulationLeasePool,
        data_manager: DataManager,
        agent_start_client: AgentStartClient,
        cfg: SimulationRunConfig,
    ) -> None:
        self.lease_pool = lease_pool
        self.data_manager = data_manager
        self.agent_start_client = agent_start_client
        self.cfg = cfg
        self.worker_count = self._derive_worker_count()
        self._results: Dict[str, SimulationStartResult] = {}
        self._results_lock = asyncio.Lock()

    async def run_all(self) -> SimulationRunSummary:
        log.info(
            "simulation workers starting: warm_pool_size=%d workers=%d",
            self.lease_pool.pool_size,
            self.worker_count,
        )
        tasks = [
            asyncio.create_task(self._worker_loop(worker_id), name=f"simulation-worker-{worker_id}")
            for worker_id in range(self.worker_count)
        ]
        cancelled = False
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            cancelled = True
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        except Exception:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        async with self._results_lock:
            results = dict(self._results)

        if not results:
            return SimulationRunSummary(
                job_id=self.cfg.job_id,
                status="failed_no_episodes",
                total_episodes=0,
                succeeded_episodes=0,
                failed_episodes=0,
                cancelled=cancelled,
                results={},
            )

        succeeded = sum(1 for result in results.values() if result.status == "succeeded")
        failed = len(results) - succeeded
        status = "cancelled" if cancelled else ("succeeded" if failed == 0 else "completed_with_failures")
        return SimulationRunSummary(
            job_id=self.cfg.job_id,
            status=status,
            total_episodes=len(results),
            succeeded_episodes=succeeded,
            failed_episodes=failed,
            cancelled=cancelled,
            results={key: result.total_reward for key, result in results.items()},
        )

    async def _worker_loop(self, worker_id: int) -> None:
        while True:
            lease = await self.lease_pool.acquire()
            if lease is None:
                log.info("worker=%d: lease pool exhausted", worker_id)
                return

            agent_key = f"{lease.agent_name}_{lease.agent_id}"
            started = time.perf_counter()
            try:
                log.info(
                    "worker=%d acquired agent=%s container=%s reuse=%s",
                    worker_id,
                    agent_key,
                    lease.container_name or lease.container_id,
                    lease.reuse_container,
                )
                result = await self._run_one_episode(lease, worker_id)
                async with self._results_lock:
                    self._results[agent_key] = result
            except Exception as exc:
                log.warning("worker=%d agent=%s failed before summary: %s", worker_id, agent_key, exc, exc_info=True)
                result = SimulationStartResult(
                    session_id=lease.agent_id,
                    status="failed",
                    total_reward=0.0,
                    step_count=0,
                    terminated=True,
                    truncated=False,
                    error_text=str(exc),
                    metrics={},
                )
                async with self._results_lock:
                    self._results[agent_key] = result
            finally:
                try:
                    await self.lease_pool.done(lease, result)
                except Exception:
                    log.exception("worker=%d agent=%s critical error in lease_pool.done()", worker_id, agent_key)

            elapsed = time.perf_counter() - started
            log.info(
                "worker=%d agent=%s finished status=%s reward=%.6f time=%.2fs",
                worker_id,
                agent_key,
                result.status,
                result.total_reward,
                elapsed,
            )

    async def _run_one_episode(self, lease: SimulationAgentLease, worker_id: int) -> SimulationStartResult:
        session = await self._create_session(lease)
        request = self._build_start_request(lease, session, worker_id)
        try:
            result = await self.agent_start_client.start(lease, request)
        except Exception as exc:
            log.warning(
                "worker=%d agent=%s/%s OpenClaw start failed: %s",
                worker_id,
                lease.agent_name,
                lease.agent_id,
                exc,
            )
            result = SimulationStartResult(
                session_id=session.session_id,
                status="failed",
                total_reward=0.0,
                step_count=0,
                terminated=True,
                truncated=False,
                error_text=str(exc),
                metrics={},
            )

        if result.status != "succeeded":
            log.warning(
                "worker=%d agent=%s/%s returned status=%s error=%s",
                worker_id,
                lease.agent_name,
                lease.agent_id,
                result.status,
                self._tail(result.error_text or ""),
            )

        await self._record_episode_summary(lease, session, result, worker_id)
        return result

    def _build_start_request(
        self,
        lease: SimulationAgentLease,
        session: SessionContext,
        worker_id: int,
    ) -> SimulationStartRequest:
        task_id = str(lease.row_id if lease.row_id is not None else lease.group_id or lease.agent_id)
        return SimulationStartRequest(
            job_id=self.cfg.job_id,
            task_id=task_id,
            session_id=session.session_id,
            group_id=lease.group_id,
            gateway_base_url=self.cfg.gateway_base_url,
            model=self.cfg.llm_model,
            temperature=self.cfg.llm_temperature,
            max_steps=self.cfg.max_steps,
            storage_type=self.cfg.storage_type,
            env_params=dict(lease.env_params or {}),
            storage_config={"db_url": self.cfg.db_url},
            agent_start_timeout_s=self.cfg.agent_start_timeout_s,
            record_mode="agent_runtime",
            agent_name=lease.agent_name,
            agent_id=lease.agent_id,
            metadata={
                "worker_id": worker_id,
                "row_id": lease.row_id,
                "image": lease.image,
                "container_id": lease.container_id,
                "container_name": lease.container_name,
                "reuse_container": lease.reuse_container,
                "agent_name": lease.agent_name,
                "agent_id": lease.agent_id,
                "env_params": lease.env_params,
            },
        )

    async def _record_episode_summary(
        self,
        lease: SimulationAgentLease,
        session: SessionContext,
        result: SimulationStartResult,
        worker_id: int,
    ) -> None:
        env_state = {
            "event_type": "episode_summary",
            "job_id": self.cfg.job_id,
            "worker_id": worker_id,
            "agent_name": lease.agent_name,
            "agent_id": lease.agent_id,
            "group_id": lease.group_id,
            "row_id": lease.row_id,
            "start_result": asdict(result),
        }
        try:
            await self.data_manager.record_step(
                session=session,
                step_id=max(1, int(result.step_count or 0) + 1),
                messages=[],
                response=result.error_text or result.status,
                step_reward=0.0,
                env_state=json.dumps(env_state, ensure_ascii=False, default=str),
                terminated=result.terminated,
                truncated=result.truncated,
                is_trainable=False,
            )
        except Exception:
            log.exception("failed to record episode summary for %s/%s", lease.agent_name, lease.agent_id)

    async def _create_session(self, lease: SimulationAgentLease) -> SessionContext:
        maybe_session = self.data_manager.create_session(
            env_id=lease.agent_id,
            env_name=lease.agent_name,
            llm_model=self.cfg.llm_model,
            group_id=lease.group_id,
            job_id=self.cfg.job_id,
        )
        if inspect.isawaitable(maybe_session):
            return await maybe_session
        return maybe_session

    def _derive_worker_count(self) -> int:
        worker_count = max(1, int(getattr(self.lease_pool, "pool_size", 1) or 1))
        if self.cfg.max_workers is not None:
            worker_count = max(1, min(worker_count, int(self.cfg.max_workers)))
        return worker_count

    @staticmethod
    def _tail(value: str, limit: int = 1000) -> str:
        return (value or "").strip()[-int(limit):]

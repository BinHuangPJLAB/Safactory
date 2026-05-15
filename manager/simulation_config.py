from __future__ import annotations

import argparse
import logging
import math
import os
import uuid
from collections.abc import Sequence
from typing import Any, Dict, List, Tuple

import yaml

from .types import SimulationRunConfig

log = logging.getLogger("manager.simulation_config")


def parse_simulation_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safactory task-level simulation launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--job-id", type=str, default="", help="Simulation workflow id")
    parser.add_argument("--manager-config", type=str, default="./manager/config.yaml")
    parser.add_argument("--exp-config", type=str, default="./core/exp/config.yaml")
    parser.add_argument("--mode", choices=["docker"], default="docker")

    parser.add_argument("--agent-config", type=str, default=None, help="Single agent YAML config")
    parser.add_argument("--agent-root", type=str, default="env", help="Directory scanned for agent YAML configs")
    parser.add_argument("--storage-type", type=str, default="sqlite", choices=["sqlite", "cloud"])
    parser.add_argument("--db-path", type=str, default="sqlite://env_trajs.db")
    parser.add_argument("--rebuild-table", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-buffer", dest="enable_buffer", action="store_false", default=True)
    parser.add_argument("--buffer-size", type=int, default=100)
    parser.add_argument("--flush-interval", type=float, default=5.0)

    parser.add_argument("--pool-size", type=int, default=0, help="Override manager pool_size from YAML")
    parser.add_argument("--multiplier", type=float, default=1.2, help="Warm-pool multiplier")
    parser.add_argument("--max-workers", type=int, default=0, help="0 means use warm-pool size")

    parser.add_argument("--gateway-base-url", type=str, default="http://127.0.0.1:8080/v1/sessions")
    parser.add_argument("--agent-start-timeout-s", type=float, default=3600.0)
    parser.add_argument("--agent-runtime", choices=["agent_start"], default="agent_start")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--llm-model", type=str, default="default")
    parser.add_argument("--llm-temperature", type=float, default=0.3)
    parser.add_argument("--rl-group-size", type=int, default=0)
    parser.add_argument("--rl-epoch", type=int, default=1)

    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--console-log-level", type=str, default="INFO")
    parser.add_argument("--file-log-level", type=str, default="DEBUG")
    parser.add_argument("--log-max-bytes", type=int, default=50 * 1024 * 1024)
    parser.add_argument("--log-backup-count", type=int, default=20)
    parser.add_argument("--debug-log", action="store_true", default=False)
    return parser.parse_args(argv)


def load_simulation_run_config(args: argparse.Namespace) -> SimulationRunConfig:
    manager_cfg = load_yaml_file(str(args.manager_config))
    configured_pool_size = int(manager_cfg.get("pool_size", 1) or 1)
    pool_size, warm_pool_size, startup_submit_count, followup_submit_batch = derive_pool_sizing(
        configured_pool_size=configured_pool_size,
        pool_size_override=int(args.pool_size),
        multiplier=float(args.multiplier),
    )

    mode = str(args.mode or manager_cfg.get("mode") or "docker").strip().lower()
    if mode != "docker":
        raise ValueError(f"Only docker mode is supported by the OpenClaw workflow; got {mode!r}")

    job_id = str(args.job_id or "").strip() or uuid.uuid4().hex
    max_workers = int(args.max_workers) if int(args.max_workers or 0) > 0 else None

    _validate_gateway_route_key(str(args.llm_model))
    agent_config = getattr(args, "agent_config", None)

    return SimulationRunConfig(
        job_id=job_id,
        manager_config_path=str(args.manager_config),
        exp_config_path=str(args.exp_config),
        agent_root=str(args.agent_root),
        agent_config=None if agent_config is None else str(agent_config),
        storage_type=str(args.storage_type),
        db_url=str(args.db_path),
        pool_size=pool_size,
        warm_pool_size=warm_pool_size,
        startup_submit_count=startup_submit_count,
        followup_submit_batch=followup_submit_batch,
        mode=mode,
        gateway_base_url=str(args.gateway_base_url).rstrip("/"),
        llm_model=str(args.llm_model),
        llm_temperature=float(args.llm_temperature),
        max_steps=int(args.max_steps),
        agent_start_timeout_s=float(args.agent_start_timeout_s),
        max_workers=max_workers,
        agent_runtime=str(args.agent_runtime),
        rebuild_table=bool(args.rebuild_table),
        enable_buffer=bool(args.enable_buffer),
        buffer_size=int(args.buffer_size),
        flush_interval=float(args.flush_interval),
        rl_group_size=int(args.rl_group_size),
        rl_epoch=max(1, int(args.rl_epoch)),
    )


def derive_pool_sizing(
    configured_pool_size: int,
    pool_size_override: int,
    multiplier: float,
) -> Tuple[int, int, int, int]:
    base_pool_size = int(configured_pool_size or 1)
    if int(pool_size_override) > 0:
        base_pool_size = int(pool_size_override)

    normalized_multiplier = float(multiplier) if float(multiplier) > 0.0 else 1.2
    warm_pool_size = math.ceil(base_pool_size * normalized_multiplier)
    startup_submit_count = max(base_pool_size * 2, warm_pool_size)
    followup_submit_batch = max(1, base_pool_size)
    return base_pool_size, warm_pool_size, startup_submit_count, followup_submit_batch


def build_manager_runtime_config(cfg: SimulationRunConfig) -> Dict[str, Any]:
    manager_cfg = load_yaml_file(cfg.manager_config_path)
    manager_cfg["pool_size"] = int(cfg.warm_pool_size)
    manager_cfg["mode"] = cfg.mode
    set_nested(manager_cfg, ["database", "driver"], manager_cfg.get("database", {}).get("driver", "sqlite"))
    set_nested(manager_cfg, ["database", "sqlite_path"], cfg.db_url)

    return manager_cfg


def expand_rl_group_size(yaml_config_list: List[Dict[str, Any]], group_size: int) -> List[Dict[str, Any]]:
    if int(group_size) <= 0:
        return yaml_config_list
    expanded = [dict(item) for item in yaml_config_list]
    for item in expanded:
        item["env_num"] = int(group_size)
    log.info("Override agent parallelism env_num=%d for %d config(s)", int(group_size), len(expanded))
    return expanded


def expand_rl_epoch(yaml_config_list: List[Dict[str, Any]], epoch: int) -> List[Dict[str, Any]]:
    epoch = max(1, int(epoch))
    if epoch <= 1:
        return yaml_config_list

    expanded = list(yaml_config_list)
    base_configs = list(yaml_config_list)
    num_tasks = len(base_configs)
    for epoch_idx in range(1, epoch):
        for item in base_configs:
            epoch_item = dict(item)
            epoch_item["task_idx"] = item.get("task_idx", 1) + epoch_idx * num_tasks
            expanded.append(epoch_item)
    log.info("rl_epoch=%d: expanded %d configs to %d configs", epoch, num_tasks, len(expanded))
    return expanded


def load_yaml_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid yaml root (expected dict): {path}")
    return cfg


def set_nested(cfg: Dict[str, Any], path: List[str], value: Any) -> None:
    cur: Dict[str, Any] = cfg
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def rebuild_sqlite_db(db_url: str) -> None:
    if not db_url.startswith("sqlite://"):
        return
    file_path = db_url[len("sqlite://") :].split("?", 1)[0]
    if file_path and os.path.exists(file_path):
        os.remove(file_path)
        log.info("Removed existing SQLite DB for rebuild: %s", file_path)


def _validate_gateway_route_key(model: str) -> None:
    normalized_model = str(model or "").strip()
    placeholder_models = {
        "YOUR_ROUTE",
        "YOUR_MODEL",
        "YOUR_GATEWAY_ROUTE_KEY",
        "YOUR_LLM_MODEL",
    }
    if normalized_model.upper() in placeholder_models:
        raise ValueError(
            f"--llm-model must be a real gateway llm_routes key, got placeholder {model!r}"
        )

    gateway_config_path = os.environ.get("AIEVOBOX_GATEWAY_CONFIG")
    if not gateway_config_path:
        log.debug("AIEVOBOX_GATEWAY_CONFIG is unset; skip gateway route-key validation")
        return

    try:
        from gateway.config import load_gateway_config
    except Exception:
        log.debug("gateway config module is unavailable; skip route-key validation", exc_info=True)
        return

    try:
        gateway_cfg = load_gateway_config(gateway_config_path)
    except Exception:
        log.debug("gateway config could not be loaded; skip route-key validation", exc_info=True)
        return

    routes = gateway_cfg.llm_routes or {}
    if routes and model not in routes:
        raise ValueError(
            f"--llm-model must be a gateway llm_routes key; got {model!r}, available={sorted(routes)}"
        )

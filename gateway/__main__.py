from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn

from gateway.app import create_app
from gateway.config import load_gateway_config


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AIEvo API Gateway.")
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to the gateway YAML/JSON config file. All runtime settings "
            "such as listen address, storage, telemetry, and LLM routes are "
            "loaded from this file."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = load_gateway_config(args.config)
    uvicorn.run(
        create_app(cfg),
        host=cfg.listen_host,
        port=cfg.listen_port,
        reload=False,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from app import create_app
from config import RuntimeSettings


def main() -> None:
    load_dotenv()
    runtime = RuntimeSettings()
    parser = argparse.ArgumentParser(prog="inference-server")
    parser.add_argument("--config", default=str(runtime.config_path))
    parser.add_argument("--host", default=runtime.host)
    parser.add_argument("--port", type=int, default=runtime.port)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    runtime.config_path = Path(args.config)

    def app_factory() -> FastAPI:
        return create_app(runtime=runtime)

    uvicorn.run(
        app_factory,
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()

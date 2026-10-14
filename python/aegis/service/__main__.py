"""Run the Aegis API server: ``python -m aegis.service [--host 127.0.0.1] [--port 8000]``."""

import argparse
import logging

import uvicorn

from .app import create_app

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Loopback by default: exposing the server to the network is a deliberate choice.
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(f"\n  Aegis dashboard: http://{args.host}:{args.port}\n")
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")

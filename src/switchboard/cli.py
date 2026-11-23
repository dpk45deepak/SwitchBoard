"""Command line: ``switchboard serve | worker | keys create | submit``."""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys

from .config import Settings
from .observability.logging import setup_logging


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings.from_env()
    if getattr(args, "db", None):
        s.db_path = args.db
    return s


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn  # needs the 'fastapi' + 'uvicorn' dependencies

    if args.db:
        import os
        os.environ["SWITCHBOARD_DB"] = args.db
    if args.with_worker:
        import os
        os.environ["EMBEDDED_WORKER"] = "true"
    uvicorn.run("switchboard.api.app:create_app", factory=True, host=args.host, port=args.port)


def cmd_worker(args: argparse.Namespace) -> None:
    from .gateway import build_gateway
    from .scheduler.queue import TaskQueue
    from .scheduler.worker import Worker

    s = _settings(args)
    setup_logging(s.log_level, s.log_redact_pii)
    gw = build_gateway(s)
    w = Worker(TaskQueue(gw.db, gw.clock), gw, concurrency=args.concurrency, lease_s=args.lease_seconds,  # type: ignore[attr-defined]
               max_attempts=args.max_attempts)

    async def main() -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):  # graceful shutdown: finish or release in-flight tasks
            loop.add_signal_handler(sig, w.stop)
        await w.run(until_idle=args.until_idle, grace_s=args.grace_seconds)
        print(json.dumps({"worker": w.owner, **w.stats}))

    asyncio.run(main())


def cmd_keys(args: argparse.Namespace) -> None:
    from .db.session import Database
    from .clock import Clock
    from .ledger.budgets import KeyStore

    s = _settings(args)
    ks = KeyStore(Database(s.db_path), Clock())
    token, key = ks.create(args.name, args.rpm, args.tpm, args.daily_budget, downgrade=args.downgrade)
    print(json.dumps({"id": key.id, "api_key": token, "note": "store this token now; only its hash is kept"}))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="switchboard")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("serve")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--db")
    sp.add_argument("--with-worker", action="store_true", help="run a batch worker inside the API process (lite mode)")
    sp.set_defaults(fn=cmd_serve)
    wp = sub.add_parser("worker")
    wp.add_argument("--db")
    wp.add_argument("--concurrency", type=int, default=16)
    wp.add_argument("--lease-seconds", type=float, default=30.0)
    wp.add_argument("--max-attempts", type=int, default=5)
    wp.add_argument("--grace-seconds", type=float, default=10.0)
    wp.add_argument("--until-idle", action="store_true")
    wp.set_defaults(fn=cmd_worker)
    kp = sub.add_parser("keys")
    kp.add_argument("action", choices=["create"])
    kp.add_argument("--db")
    kp.add_argument("--name", default="")
    kp.add_argument("--rpm", type=float)
    kp.add_argument("--tpm", type=float)
    kp.add_argument("--daily-budget", type=float)
    kp.add_argument("--downgrade", action="store_true")
    kp.set_defaults(fn=cmd_keys)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])

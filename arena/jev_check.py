"""Connectivity check for the Jev / OpenRouter decisions API.

    python -m arena.jev_check              # 3 decisions on a real observation
    python -m arena.jev_check --show-request

Reads OPENROUTER_API_KEY (and optional JEV_*) from .env. Prints round-trip
latency, the parsed action and the raw reply for each call, so response
format problems are visible right away.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from arena.action import Action
from arena.difficulty import PRESETS
from arena.dotenv import load_dotenv
from arena.environment import Environment


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=3, help="number of requests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show-request", action="store_true", help="print the full request body")
    args = ap.parse_args(argv)

    env_file = load_dotenv()
    from controllers.jev import JevController

    try:
        ctrl = JevController()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 2
    print(f".env:     {env_file or 'not found (using process environment)'}")
    print(f"endpoint: {ctrl.endpoint}")
    print(f"model:    {ctrl.model}")

    env = Environment(PRESETS["medium"], args.seed)
    ctrl.reset(env.public_info())
    ok = 0
    for i in range(args.n):
        body = ctrl.encode_decide(i, env.observe())
        if args.show_request and i == 0:
            print("\nrequest body:\n" + json.dumps(body, indent=2))
        t0 = time.perf_counter()
        try:
            payload = ctrl._post(ctrl.decide_path, body)
        except Exception as e:
            print(f"\n[{i}] FAILED after {(time.perf_counter() - t0) * 1000:.0f} ms: {type(e).__name__}: {e}")
            continue
        ms = (time.perf_counter() - t0) * 1000
        try:
            action, meta = ctrl.decode_decide(payload, i)
            ok += 1
            conf = (meta or {}).get("confidence")
            extra = f"  confidence={conf:.2f}" if isinstance(conf, (int, float)) else ""
            layout = "  (fallback layout, not the documented one)" if (meta or {}).get("layout") else ""
            print(f"\n[{i}] {ms:.0f} ms  action={action.value}{extra}{layout}")
        except ValueError as e:
            print(f"\n[{i}] {ms:.0f} ms  UNPARSEABLE: {e}")
        print("raw reply: " + json.dumps(payload)[:2000])
        for _ in range(6):  # let the world move a little between calls
            env.step(Action.STAY)
    ctrl.close()
    print(f"\n{ok}/{args.n} decisions parsed")
    return 0 if ok == args.n else 1


if __name__ == "__main__":
    sys.exit(main())

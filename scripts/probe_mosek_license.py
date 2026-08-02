#!/usr/bin/env python
"""Diagnose why `is_mosek_available()` costs ~70s per call on macOS arm64 CI.

On `macos-latest` runners each call to `ncpoleon.utils.is_mosek_available()` blocks for
70.1s; on ubuntu-latest and windows-latest the same call returns in ~70ms. The duration is
identical to three significant figures across independent runs, so it is a fixed timeout
rather than work, but which timeout is not known. This script separates the candidates.

Two hypotheses are on the table:

1. The license system is trying to reach a license server and waiting out a connect
   timeout. MOSEK documents a ~60s checkout timeout, but its documented default search
   path is local-only (`$HOME/mosek/mosek.lic`), so there should be no server to contact
   unless something sets `MOSEKLM_LICENSE_FILE`.
2. Host identification resolves the machine's own hostname, and GitHub's macOS runners
   have `.local` hostnames that do not resolve, so the lookup waits out mDNSResponder.

Phase 3 tests hypothesis 2 without involving MOSEK at all; phases 4-6 test hypothesis 1
using MOSEK's own license tracing. Read the per-phase timings first -- whichever phase
holds the ~70s is the answer, and the trace only explains it if the time is in phase 6.

`putlicensedebug` can only be enabled on an already-constructed `Env`, so it cannot
instrument the constructor. That is why phase 5 is timed separately from phase 6: if the
stall is in `Env()` itself, the trace will be silent and the timings are the only evidence.

Run under `python -u`; a 70s stall behind a block-buffered stdout reorders the CI log
badly enough to be misleading.

Usage:
    uv run --no-project --with mosek python -u scripts/probe_mosek_license.py
"""

from __future__ import annotations

import os
import socket
import sys
import time

FEATURE = "pts"


def _report(label: str, started: float, outcome: object) -> None:
    print(f"[{time.perf_counter() - started:7.2f}s] {label}: {outcome}", flush=True)


def _timed(label: str, call) -> None:
    started = time.perf_counter()
    try:
        outcome = call()
    except Exception as exc:  # noqa: BLE001 - a raising probe is a result, not a failure
        outcome = f"{type(exc).__name__}: {exc}"
    _report(label, started, outcome)


def main() -> int:
    print("=== phase 1: environment ===", flush=True)
    for name in ("MOSEKLM_LICENSE_FILE", "MOSEK_LICENSE_FILE", "LM_LICENSE_FILE"):
        print(f"{name}={os.environ.get(name)!r}", flush=True)
    home = os.path.expanduser("~")
    default_license = os.path.join(home, "mosek", "mosek.lic")
    print(f"HOME={home!r}", flush=True)
    print(f"{default_license} exists: {os.path.exists(default_license)}", flush=True)

    print("\n=== phase 2: platform ===", flush=True)
    print(f"{sys.platform} {os.uname().machine} python {sys.version.split()[0]}", flush=True)

    print("\n=== phase 3: hostname resolution (no MOSEK involved) ===", flush=True)
    hostname = socket.gethostname()
    print(f"gethostname()={hostname!r}", flush=True)
    _timed("gethostbyname(hostname)", lambda: socket.gethostbyname(hostname))
    # Only the timing matters here, and the full address list runs to dozens of tuples.
    _timed("getaddrinfo(hostname)", lambda: f"{len(socket.getaddrinfo(hostname, None))} addresses")
    _timed("getfqdn()", socket.getfqdn)

    print("\n=== phase 4: import mosek ===", flush=True)
    started = time.perf_counter()
    import mosek

    _report("import mosek", started, mosek.Env.getversion())

    print("\n=== phase 5: Env() construction (cannot be traced, see docstring) ===", flush=True)
    started = time.perf_counter()
    env = mosek.Env()
    _report("mosek.Env()", started, "constructed")

    print("\n=== phase 6: checkoutlicense with license debug on ===", flush=True)
    env.set_Stream(mosek.streamtype.log, lambda text: (sys.stdout.write(text), sys.stdout.flush()))
    env.putlicensedebug(1)
    _timed("checkoutlicense", lambda: env.checkoutlicense(getattr(mosek.feature, FEATURE)))

    # is_mosek_available() pays the cost once per call, so the question of whether the
    # timeout is per-process or per-Env decides whether pinning the search path is even
    # worth trying. A fast second cycle would mean something is cached and the 21 calls
    # are paying for Env churn instead.
    print("\n=== phase 7: second Env + checkout, same process ===", flush=True)
    started = time.perf_counter()
    second = mosek.Env()
    _report("mosek.Env() again", started, "constructed")
    second.set_Stream(mosek.streamtype.log, lambda text: (sys.stdout.write(text), sys.stdout.flush()))
    second.putlicensedebug(1)
    _timed("checkoutlicense again", lambda: second.checkoutlicense(getattr(mosek.feature, FEATURE)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""sys-pulse: terminal process / port / GPU / AI-runtime monitor.

  python -m syspulse                      interactive TUI
  python -m syspulse --once --sort mem    one snapshot table and exit
  python -m syspulse --once --ports       listening ports by process
  python -m syspulse --once --ai --json   AI/ML runtimes as JSON
  python -m syspulse --once -f "python cpu>1"
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time

from .collector import Collector
from .model import SORT_ORDER
from .tui import State, render, run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="syspulse", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit")
    ap.add_argument("--json", action="store_true", help="with --once: JSON output")
    ap.add_argument("-n", "--rows", type=int, default=25)
    ap.add_argument("-s", "--sort", choices=SORT_ORDER, default="cpu")
    ap.add_argument("-f", "--filter", default="")
    ap.add_argument("-t", "--tree", action="store_true")
    ap.add_argument("--ports", action="store_true")
    ap.add_argument("--ai", action="store_true")
    ap.add_argument("-i", "--interval", type=float, default=1.5)
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    state = State(sort=a.sort, tree=a.tree, view="ports" if a.ports else "procs", ai_only=a.ai, filter=a.filter)
    col = Collector()
    if not a.once:
        if not sys.stdin.isatty():
            print("sys-pulse: interactive mode needs a terminal; use --once", file=sys.stderr)
            return 2
        return run(col, a.interval, state)

    col.sample()
    time.sleep(max(0.2, min(a.interval, 1.0)))  # rates need two samples
    procs, st = col.sample()
    if a.json:
        from .tui import select_rows
        rows = select_rows(state, procs)[: a.rows]
        print(json.dumps({"cpu_percent": round(st.cpu_percent, 1), "mem_used": st.mem_used,
                          "mem_total": st.mem_total, "gpus": st.gpus, "sample_ms": round(col.last_sample_ms, 1),
                          "processes": [{"pid": p.pid, "ppid": p.ppid, "name": p.name, "cpu": round(p.cpu, 1),
                                         "rss": p.rss, "read_bps": round(p.read_bps), "write_bps": round(p.write_bps),
                                         "gpu_mem": p.gpu_mem, "threads": p.threads, "tags": p.tags,
                                         "ports": [list(x) for x in p.ports], "cmdline": p.cmdline}
                                        for p in rows]}, indent=2))
        return 0
    width = shutil.get_terminal_size((130, 40)).columns if sys.stdout.isatty() else 130
    color = sys.stdout.isatty() and not a.no_color
    frame = render(state, procs, st, width, 4 + len(st.gpus) + 1 + a.rows + 1, color,
                   col.last_sample_ms, col.b.name)
    print("\n".join(line.rstrip() for line in frame[:-1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

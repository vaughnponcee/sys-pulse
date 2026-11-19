"""Platform-independent data model: raw samples -> rates, trees, filters, sorting."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class RawProc:
    """What a backend reports. Times/bytes are cumulative counters."""
    pid: int
    ppid: int
    name: str
    cpu_time: float = 0.0          # user+kernel seconds since process start
    rss: int = 0                   # bytes
    io_read: int = 0               # cumulative bytes
    io_write: int = 0
    threads: int = 0
    cmdline: str = ""
    user: str = ""
    start_time: float = 0.0        # used to detect PID reuse


@dataclass
class Proc:
    pid: int
    ppid: int
    name: str
    cpu: float                     # percent of ONE core (top-style; can exceed 100)
    rss: int
    read_bps: float
    write_bps: float
    threads: int
    cmdline: str
    ports: list = field(default_factory=list)   # [(proto, addr, port)]
    gpu_mem: int = 0
    tags: list = field(default_factory=list)    # e.g. ["ai:ollama", "python"]
    depth: int = 0                              # tree view indentation


@dataclass
class SystemStats:
    cpu_percent: float = 0.0        # of all cores, 0..100
    cores: int = 1
    mem_total: int = 0
    mem_used: int = 0
    gpus: list = field(default_factory=list)    # [{"name","util","mem_used","mem_total","temp"}]


def compute(prev: dict[int, RawProc] | None, cur: dict[int, RawProc], dt: float) -> list[Proc]:
    """Turn two samples `dt` seconds apart into per-process rates. A PID whose
    start_time changed between samples was reused: treat it as new (0 rate)."""
    out = []
    for pid, r in cur.items():
        p = (prev or {}).get(pid)
        if p is not None and p.start_time == r.start_time and dt > 0:
            cpu = max(0.0, (r.cpu_time - p.cpu_time) / dt * 100)
            rd = max(0.0, (r.io_read - p.io_read) / dt)
            wr = max(0.0, (r.io_write - p.io_write) / dt)
        else:
            cpu = rd = wr = 0.0
        out.append(Proc(r.pid, r.ppid, r.name, cpu, r.rss, rd, wr, r.threads, r.cmdline))
    return out


def tree_order(procs: list[Proc], key=None) -> list[Proc]:
    """Depth-first parent->children order with `depth` set. Processes whose
    parent is gone (or that form a ppid cycle) become roots."""
    by_pid = {p.pid: p for p in procs}
    kids: dict[int, list[Proc]] = {}
    roots = []
    for p in procs:
        parent = by_pid.get(p.ppid)
        if parent is None or p.ppid == p.pid or _in_cycle(p, by_pid):
            roots.append(p)
        else:
            kids.setdefault(p.ppid, []).append(p)
    key = key or (lambda p: p.pid)
    out: list[Proc] = []
    seen: set[int] = set()

    def walk(p: Proc, depth: int):
        if p.pid in seen:
            return
        seen.add(p.pid)
        p.depth = depth
        out.append(p)
        for c in sorted(kids.get(p.pid, []), key=key):
            walk(c, depth + 1)

    for r in sorted(roots, key=key):
        walk(r, 0)
    return out


def _in_cycle(p: Proc, by_pid: dict[int, Proc]) -> bool:
    seen = {p.pid}
    cur = by_pid.get(p.ppid)
    while cur is not None:
        if cur.pid in seen:
            return True
        seen.add(cur.pid)
        cur = by_pid.get(cur.ppid)
    return False


def subtree(procs: list[Proc], root: int) -> list[Proc]:
    kids: dict[int, list[Proc]] = {}
    for p in procs:
        kids.setdefault(p.ppid, []).append(p)
    out, stack = [], [root]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        out.extend(p for p in procs if p.pid == pid)
        stack.extend(c.pid for c in kids.get(pid, []) if c.pid != pid)
    return out


# ------------------------------------------------------------------ filtering
SIZE = {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}
_TERM = re.compile(r"^(cpu|mem|rss|io|gpu|pid|ppid|port|threads)\s*(>=|<=|>|<|=|:)\s*([\w.]+)$", re.I)


def _num(v: str) -> float:
    m = re.fullmatch(r"([\d.]+)\s*([kmg]?)b?", v.lower())
    if not m:
        raise ValueError(f"bad number {v!r}")
    return float(m.group(1)) * SIZE[m.group(2)]


def parse_filter(expr: str):
    """Vim-style `/` filter. Space-separated terms, all must match:
      python          substring of name or command line (case-insensitive)
      !svchost        negation
      cpu>5  mem>500M  io>1M  gpu>0  threads>50  pid=1234  ppid:4  port:8080
      ai              tagged as an AI/ML runtime     listen  has a listening port
    Returns a predicate(Proc) -> bool. Raises ValueError on a malformed term."""
    preds = []
    for raw in expr.split():
        neg = raw.startswith("!")
        term = raw[1:] if neg else raw
        m = _TERM.match(term)
        if m:
            field_, op, val = m.group(1).lower(), m.group(2), m.group(3)
            if field_ == "port":
                port = int(val)
                pred = (lambda port: lambda p: any(pt == port for _, _, pt in p.ports))(port)
            else:
                num = _num(val)
                get = {"cpu": lambda p: p.cpu, "mem": lambda p: p.rss, "rss": lambda p: p.rss,
                       "io": lambda p: p.read_bps + p.write_bps, "gpu": lambda p: p.gpu_mem,
                       "pid": lambda p: p.pid, "ppid": lambda p: p.ppid, "threads": lambda p: p.threads}[field_]
                cmp = {">": float.__gt__, "<": float.__lt__, ">=": float.__ge__, "<=": float.__le__,
                       "=": float.__eq__, ":": float.__eq__}[op]
                pred = (lambda get, cmp, num: lambda p: cmp(float(get(p)), num))(get, cmp, num)
        elif term.lower() == "ai":
            pred = lambda p: any(t.startswith("ai") for t in p.tags)
        elif term.lower() == "listen":
            pred = lambda p: bool(p.ports)
        elif re.match(r"^(cpu|mem|rss|io|gpu|pid|ppid|port|threads)[<>=:]", term, re.I):
            raise ValueError(f"bad filter term {term!r}")
        else:
            needle = term.lower()
            pred = (lambda needle: lambda p: needle in p.name.lower() or needle in p.cmdline.lower())(needle)
        preds.append((lambda pred: (lambda p: not pred(p)))(pred) if neg else pred)
    return lambda p: all(f(p) for f in preds)


SORT_KEYS = {
    "cpu": lambda p: (-p.cpu, p.pid),
    "mem": lambda p: (-p.rss, p.pid),
    "io": lambda p: (-(p.read_bps + p.write_bps), p.pid),
    "gpu": lambda p: (-p.gpu_mem, -p.cpu, p.pid),
    "pid": lambda p: p.pid,
    "name": lambda p: (p.name.lower(), p.pid),
}
SORT_ORDER = list(SORT_KEYS)

"""ANSI terminal UI. `render()` is a pure function of state + data (tested);
`run()` owns the terminal and the key loop."""
from __future__ import annotations

import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field

from .model import SORT_KEYS, SORT_ORDER, Proc, SystemStats, parse_filter, subtree, tree_order

CSI = "\033["
ANSI = re.compile(r"\033\[[0-9;?]*[A-Za-z]")
HELP = [
    "j/k or arrows  move          g/G   top/bottom      PgUp/PgDn  page",
    "/              filter        Esc   clear filter     s  cycle sort (cpu mem io gpu pid name)",
    "t              tree view     p     ports view       a  AI/GPU runtimes only",
    "Enter          focus subtree of selected process   Backspace  leave focus",
    "space          pause         ?     this help        q  quit",
    "",
    "filter terms (all must match): python  !svchost  cpu>5  mem>500M  io>1M  gpu>0",
    "                               port:8080  pid=1234  threads>50  ai  listen",
]


@dataclass
class State:
    sort: str = "cpu"
    tree: bool = False
    view: str = "procs"          # procs | ports
    ai_only: bool = False
    filter: str = ""
    filter_error: str = ""
    editing: bool = False
    edit_buf: str = ""
    sel: int = 0
    top: int = 0
    focus: int | None = None
    paused: bool = False
    help: bool = False
    rows: list = field(default_factory=list)


def human(n: float, unit: str = "B") -> str:
    for u in ("", "K", "M", "G", "T"):
        if abs(n) < 1024 or u == "T":
            return f"{n:.0f}{u}{unit}" if u == "" else f"{n:.1f}{u}{unit}"
        n /= 1024
    return f"{n:.1f}T{unit}"


def bar(frac: float, width: int, color: bool = True) -> str:
    frac = max(0.0, min(1.0, frac))
    full = int(frac * width)
    s = "|" * full + " " * (width - full)
    if not color:
        return f"[{s}]"
    c = "32" if frac < 0.6 else ("33" if frac < 0.85 else "31")
    return f"[{CSI}{c}m{s}{CSI}0m]"


def visible(s: str) -> int:
    return len(ANSI.sub("", s))


def fit(s: str, width: int) -> str:
    """Pad/truncate to exactly `width` visible columns, ANSI-aware."""
    out, n, i = [], 0, 0
    while i < len(s) and n < width:
        m = ANSI.match(s, i)
        if m:
            out.append(m.group())
            i = m.end()
            continue
        out.append(s[i])
        n += 1
        i += 1
    return "".join(out) + " " * (width - n) + (f"{CSI}0m" if "\033" in s else "")


def select_rows(state: State, procs: list[Proc]) -> list[Proc]:
    rows = procs
    if state.focus is not None:
        rows = subtree(rows, state.focus)
    if state.ai_only:
        rows = [p for p in rows if any(t.startswith("ai:") for t in p.tags)]
    if state.view == "ports":
        rows = [p for p in rows if p.ports]
    state.filter_error = ""
    if state.filter:
        try:
            pred = parse_filter(state.filter)
            rows = [p for p in rows if pred(p)]
        except ValueError as e:
            state.filter_error = str(e)
    key = SORT_KEYS[state.sort]
    if state.tree and state.view == "procs":
        # keep matching rows; tree order among them
        rows = tree_order(rows, key=key)
    elif state.view == "ports":
        rows = sorted(rows, key=lambda p: (p.ports[0][2], p.pid))
    else:
        rows = sorted(rows, key=key)
    return rows


def render(state: State, procs: list[Proc], st: SystemStats, width: int, height: int,
           color: bool = True, sample_ms: float = 0.0, backend: str = "") -> list[str]:
    B = (lambda s: f"{CSI}1m{s}{CSI}0m") if color else (lambda s: s)
    DIM = (lambda s: f"{CSI}2m{s}{CSI}0m") if color else (lambda s: s)
    INV = (lambda s: f"{CSI}7m{s}{CSI}0m") if color else (lambda s: "> " + s[2:])
    lines = []
    bw = max(10, min(40, width - 40))
    lines.append(f" {B('CPU')} {bar(st.cpu_percent / 100, bw, color)} {st.cpu_percent:5.1f}%  {st.cores} cores"
                 f"   {DIM(f'{len(procs)} procs  sample {sample_ms:.0f} ms  {backend}')}")
    mf = st.mem_used / st.mem_total if st.mem_total else 0
    lines.append(f" {B('MEM')} {bar(mf, bw, color)} {human(st.mem_used)} / {human(st.mem_total)}")
    for i, g in enumerate(st.gpus):
        util = g["util"] if g["util"] is not None else 0
        temp = f"{g['temp']:.0f}C" if g.get("temp") is not None else ""
        lines.append(f" {B(f'GPU{i}')} {bar(util / 100, bw, color)} {util:5.1f}%  "
                     f"{human(g['mem_used'])} / {human(g['mem_total'])} {temp} {DIM(g['name'])}")
    rows = select_rows(state, procs)
    state.rows = rows
    mode = [state.view, f"sort:{state.sort}"] + (["tree"] if state.tree else []) \
        + (["ai-only"] if state.ai_only else []) + ([f"focus:{state.focus}"] if state.focus is not None else []) \
        + (["PAUSED"] if state.paused else [])
    lines.append(DIM(" " + "  ".join(mode) + (f"   filter: {state.filter}" if state.filter else "")))
    if state.help:
        lines += [""] + [" " + h for h in HELP]
        return [fit(ln, width) for ln in lines[:height]]

    if state.view == "ports":
        hdr = f"{'PID':>7} {'NAME':<22} {'PROTO':<5} {'ADDRESS':<28} {'PORT':>6}  TAGS"
    else:
        hdr = f"{'PID':>7} {'PPID':>7} {'CPU%':>6} {'MEM':>8} {'READ/s':>8} {'WRITE/s':>8} {'GPU':>7} {'THR':>4}  NAME / COMMAND"
    lines.append(INV(fit(hdr, width)) if color else fit(hdr, width))
    body_h = max(1, height - len(lines) - 1)
    state.sel = max(0, min(state.sel, len(rows) - 1))
    if state.sel < state.top:
        state.top = state.sel
    elif state.sel >= state.top + body_h:
        state.top = state.sel - body_h + 1
    for i, p in enumerate(rows[state.top:state.top + body_h], start=state.top):
        tags = " ".join(t for t in p.tags if t != "python")
        if state.view == "ports":
            proto, addr, port = p.ports[0]
            uniq = []
            for pr, _, pt in p.ports:  # same port on v4 + v6 / several NICs counts once
                if (pr, pt) not in uniq:
                    uniq.append((pr, pt))
            addrs = sorted({a for pr, a, pt in p.ports if (pr, pt) == (proto, port)})
            addr_s = addr if len(addrs) == 1 else f"{addr} +{len(addrs) - 1}"
            line = f"{p.pid:>7} {p.name[:22]:<22} {proto:<5} {addr_s[:28]:<28} {port:>6}  {tags}"
            if len(uniq) > 1:
                more = ", ".join(f"{pt}/{pr}" for pr, pt in uniq[1:7])
                line += DIM(f"  also {more}" + (" ..." if len(uniq) > 7 else ""))
        else:
            indent = ("  " * p.depth + ("└ " if p.depth else "")) if state.tree else ""
            cmd = p.cmdline if p.cmdline and p.cmdline != p.name else ""
            label = f"{indent}{p.name}"
            if tags:
                label += f" [{tags}]"
            if cmd:
                label += "  " + DIM(cmd)
            cpu = f"{p.cpu:6.1f}"
            if color and p.cpu >= 50:
                cpu = f"{CSI}31m{cpu}{CSI}0m"
            elif color and p.cpu >= 10:
                cpu = f"{CSI}33m{cpu}{CSI}0m"
            gpu = human(p.gpu_mem) if p.gpu_mem else "-"
            line = (f"{p.pid:>7} {p.ppid:>7} {cpu} {human(p.rss):>8} {human(p.read_bps):>8} "
                    f"{human(p.write_bps):>8} {gpu:>7} {p.threads:>4}  {label}")
        line = fit(line, width)
        if i == state.sel:
            line = INV(ANSI.sub("", line)) if color else "> " + ANSI.sub("", line)[2:]
        lines.append(line)
    while len(lines) < height - 1:
        lines.append("")
    if state.editing:
        foot = f"/{state.edit_buf}_"
    elif state.filter_error:
        foot = f"filter error: {state.filter_error}"
    else:
        foot = DIM(f"{len(rows)} shown   j/k move  / filter  s sort  t tree  p ports  a ai  Enter focus  ? help  q quit")
    lines.append(foot)
    return [fit(ln, width) for ln in lines[:height]]


# ---------------------------------------------------------------- input
class Keys:
    """Non-blocking single-key reader for Windows consoles and POSIX ttys."""

    def __enter__(self):
        if os.name == "nt":
            import msvcrt
            self._m = msvcrt
        else:
            import termios
            import tty
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        if os.name != "nt":
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def get(self, timeout: float) -> str | None:
        if os.name == "nt":
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                if self._m.kbhit():
                    c = self._m.getwch()
                    if c in ("\x00", "\xe0"):
                        return {"H": "UP", "P": "DOWN", "I": "PGUP", "Q": "PGDN", "G": "HOME", "O": "END"}.get(
                            self._m.getwch(), None)
                    return {"\r": "ENTER", "\x1b": "ESC", "\x08": "BACKSPACE"}.get(c, c)
                time.sleep(0.02)
            return None
        import select
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if not r:
            return None
        c = os.read(self._fd, 1).decode(errors="ignore")
        if c == "\x1b":
            r, _, _ = select.select([sys.stdin], [], [], 0.02)
            if not r:
                return "ESC"
            seq = os.read(self._fd, 8).decode(errors="ignore")
            return {"[A": "UP", "[B": "DOWN", "[5~": "PGUP", "[6~": "PGDN", "[H": "HOME", "[F": "END"}.get(seq)
        return {"\n": "ENTER", "\r": "ENTER", "\x7f": "BACKSPACE"}.get(c, c)


def handle_key(state: State, k: str, page: int) -> bool:
    """Mutates state; returns False to quit."""
    if state.editing:
        if k == "ENTER":
            state.filter, state.editing, state.sel = state.edit_buf.strip(), False, 0
        elif k == "ESC":
            state.editing = False
        elif k == "BACKSPACE":
            state.edit_buf = state.edit_buf[:-1]
        elif k and len(k) == 1 and k.isprintable():
            state.edit_buf += k
        return True
    if k in ("q", "Q"):
        return False
    if state.help:
        state.help = False
        return True
    n = len(state.rows)
    if k in ("j", "DOWN"):
        state.sel = min(n - 1, state.sel + 1)
    elif k in ("k", "UP"):
        state.sel = max(0, state.sel - 1)
    elif k in ("g", "HOME"):
        state.sel = 0
    elif k in ("G", "END"):
        state.sel = max(0, n - 1)
    elif k == "PGDN":
        state.sel = min(n - 1, state.sel + page)
    elif k == "PGUP":
        state.sel = max(0, state.sel - page)
    elif k == "/":
        state.editing, state.edit_buf = True, state.filter
    elif k == "ESC":
        state.filter = ""
    elif k == "s":
        state.sort = SORT_ORDER[(SORT_ORDER.index(state.sort) + 1) % len(SORT_ORDER)]
    elif k == "t":
        state.tree = not state.tree
    elif k == "p":
        state.view = "ports" if state.view == "procs" else "procs"
        state.sel = 0
    elif k == "a":
        state.ai_only = not state.ai_only
        state.sel = 0
    elif k == "ENTER" and state.rows:
        state.focus, state.tree, state.sel = state.rows[state.sel].pid, True, 0
    elif k == "BACKSPACE":
        state.focus = None
    elif k == " ":
        state.paused = not state.paused
    elif k == "?":
        state.help = True
    return True


def run(collector, interval: float = 1.5, state: State | None = None) -> int:
    state = state or State()
    out = sys.stdout
    if os.name == "nt":
        os.system("")  # enable VT processing in the Windows console
    out.write(f"{CSI}?1049h{CSI}?25l")  # alt screen, hide cursor
    procs, st = collector.sample()
    time.sleep(0.25)
    procs, st = collector.sample()
    last = time.monotonic()
    try:
        with Keys() as keys:
            while True:
                w, h = shutil.get_terminal_size((120, 40))
                frame = render(state, procs, st, w, h, True, collector.last_sample_ms, collector.b.name)
                out.write(f"{CSI}H" + "\n".join(frame))
                out.flush()
                wait = max(0.0, interval - (time.monotonic() - last))
                k = keys.get(wait)
                if k is not None and not handle_key(state, k, h - 8):
                    return 0
                if time.monotonic() - last >= interval and not state.paused:
                    procs, st = collector.sample()
                    last = time.monotonic()
    except KeyboardInterrupt:
        return 0
    finally:
        out.write(f"{CSI}?25h{CSI}?1049l")
        out.flush()

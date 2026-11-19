"""Samples a backend and turns counters into rates."""
from __future__ import annotations

import sys
import time

from . import probes
from .model import Proc, SystemStats, compute


def default_backend():
    if sys.platform == "win32":
        from .backend_win import WindowsBackend
        return WindowsBackend()
    if sys.platform.startswith("linux"):
        from .backend_linux import LinuxBackend
        return LinuxBackend()
    raise SystemExit(f"sys-pulse: no backend for {sys.platform} (supported: Windows, Linux)")


class Collector:
    def __init__(self, backend=None, gpu: probes.Gpu | None = None, ports_every: float = 5.0):
        self.b = backend or default_backend()
        self.gpu = gpu if gpu is not None else probes.Gpu()
        self.ports_every = ports_every
        self._prev = None
        self._prev_t = 0.0
        self._prev_cpu = None
        self._ports: dict = {}
        self._ports_t = -1e9
        self.last_sample_ms = 0.0

    def sample(self) -> tuple[list[Proc], SystemStats]:
        t0 = time.perf_counter()
        now = time.monotonic()
        raw = self.b.processes()
        st, cpu = self.b.system()
        if now - self._ports_t >= self.ports_every:  # socket tables change slowly; poll less
            self._ports = self.b.listening()
            self._ports_t = now
        self.gpu.poll(now)
        dt = now - self._prev_t if self._prev is not None else 0.0
        procs = compute(self._prev, raw, dt)
        if self._prev_cpu is not None:
            busy = cpu[0] - self._prev_cpu[0]
            total = cpu[1] - self._prev_cpu[1]
            st.cpu_percent = max(0.0, min(100.0, 100.0 * busy / total)) if total > 0 else 0.0
        st.gpus = self.gpu.gpus
        probes.tag(procs, self._ports, self.gpu.apps)
        self._prev, self._prev_t, self._prev_cpu = raw, now, cpu
        self.last_sample_ms = (time.perf_counter() - t0) * 1000
        return procs, st

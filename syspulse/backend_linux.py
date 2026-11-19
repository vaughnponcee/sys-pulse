"""Linux collector: parses /proc. `root` is injectable so tests can point it
at a fixture tree."""
from __future__ import annotations

import os
from pathlib import Path

from .model import RawProc, SystemStats


class LinuxBackend:
    name = "linux"

    def __init__(self, root: str = "/proc"):
        self.root = Path(root)
        self.hz = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        self.page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096

    @staticmethod
    def parse_stat(text: str) -> dict:
        """/proc/<pid>/stat. comm is parenthesised and may contain spaces and
        ')' characters, so split on the LAST ')'."""
        lp, rp = text.index("("), text.rindex(")")
        f = text[rp + 2:].split()
        return {"pid": int(text[:lp]), "comm": text[lp + 1:rp], "state": f[0], "ppid": int(f[1]),
                "utime": int(f[11]), "stime": int(f[12]), "threads": int(f[17]),
                "starttime": int(f[19]), "rss_pages": int(f[21])}

    def processes(self) -> dict[int, RawProc]:
        out = {}
        for d in self.root.iterdir():
            if not d.name.isdigit():
                continue
            try:
                s = self.parse_stat((d / "stat").read_text())
            except (OSError, ValueError, IndexError):
                continue  # exited between listdir and read
            r = RawProc(s["pid"], s["ppid"], s["comm"], (s["utime"] + s["stime"]) / self.hz,
                        s["rss_pages"] * self.page, threads=s["threads"], start_time=s["starttime"] / self.hz)
            try:
                for line in (d / "io").read_text().splitlines():
                    k, _, v = line.partition(":")
                    if k == "read_bytes":
                        r.io_read = int(v)
                    elif k == "write_bytes":
                        r.io_write = int(v)
            except OSError:
                pass  # other users' processes need root
            try:
                r.cmdline = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
            except OSError:
                pass
            out[r.pid] = r
        return out

    def system(self) -> tuple[SystemStats, tuple[int, int]]:
        first = (self.root / "stat").read_text().splitlines()[0].split()[1:]
        vals = list(map(int, first))
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
        total = sum(vals[:8])
        mem = {}
        for line in (self.root / "meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            mem[k] = int(v.split()[0]) * 1024
        st = SystemStats(cores=os.cpu_count() or 1, mem_total=mem.get("MemTotal", 0),
                         mem_used=mem.get("MemTotal", 0) - mem.get("MemAvailable", mem.get("MemFree", 0)))
        return st, (total - idle, total)

    @staticmethod
    def parse_net(text: str, proto: str, v6: bool) -> list[tuple]:
        """Rows of /proc/net/{tcp,udp}{,6}: (proto, addr, port, inode) for
        listening TCP (state 0A) and bound UDP (state 07)."""
        rows = []
        for line in text.splitlines()[1:]:
            f = line.split()
            if len(f) < 10:
                continue
            local, state, inode = f[1], f[3], int(f[9])
            if (proto == "tcp" and state != "0A") or (proto == "udp" and state != "07"):
                continue
            hexaddr, hexport = local.split(":")
            if v6:
                b = bytes.fromhex(hexaddr)
                words = [b[i:i + 4][::-1] for i in range(0, 16, 4)]  # each 32-bit word is little-endian
                import socket
                addr = "[" + socket.inet_ntop(socket.AF_INET6, b"".join(words)) + "]"
            else:
                b = bytes.fromhex(hexaddr)[::-1]
                addr = ".".join(str(x) for x in b)
            rows.append((proto, addr, int(hexport, 16), inode))
        return rows

    def socket_inodes(self, pid_dir: Path) -> set[int]:
        out = set()
        try:
            for fd in (pid_dir / "fd").iterdir():
                try:
                    t = os.readlink(fd)
                except OSError:
                    continue
                if t.startswith("socket:["):
                    out.add(int(t[8:-1]))
        except OSError:
            pass
        return out

    def listening(self) -> dict[int, list]:
        by_inode = {}
        for fname, proto, v6 in (("tcp", "tcp", False), ("tcp6", "tcp", True),
                                 ("udp", "udp", False), ("udp6", "udp", True)):
            try:
                for row in self.parse_net((self.root / "net" / fname).read_text(), proto, v6):
                    by_inode[row[3]] = row[:3]
            except OSError:
                continue
        out: dict[int, list] = {}
        if not by_inode:
            return out
        for d in self.root.iterdir():
            if d.name.isdigit():
                hits = [by_inode[i] for i in self.socket_inodes(d) if i in by_inode]
                if hits:
                    out[int(d.name)] = sorted(set(hits), key=lambda t: (t[2], t[0]))
        return out

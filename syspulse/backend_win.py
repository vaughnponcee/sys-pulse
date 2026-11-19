"""Windows collector via ctypes: Toolhelp32 (process list), GetProcessTimes,
K32GetProcessMemoryInfo, GetProcessIoCounters, NtQueryInformationProcess
(command line), GetExtendedTcpTable/UdpTable (listening ports)."""
from __future__ import annotations

import ctypes
import socket
import struct
from ctypes import wintypes as W

from .model import RawProc, SystemStats

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
ntdll = ctypes.WinDLL("ntdll")
iphlp = ctypes.WinDLL("iphlpapi")

TH32CS_SNAPPROCESS = 0x2
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
INVALID_HANDLE = ctypes.c_void_p(-1).value


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", W.DWORD), ("cntUsage", W.DWORD), ("th32ProcessID", W.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", W.DWORD), ("cntThreads", W.DWORD),
                ("th32ParentProcessID", W.DWORD), ("pcPriClassBase", W.LONG), ("dwFlags", W.DWORD),
                ("szExeFile", W.WCHAR * 260)]


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [("cb", W.DWORD), ("PageFaultCount", W.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount",
                                                    "OtherOperationCount", "ReadTransferCount",
                                                    "WriteTransferCount", "OtherTransferCount")]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", W.DWORD), ("dwMemoryLoad", W.DWORD), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


k32.CreateToolhelp32Snapshot.restype = W.HANDLE
k32.CreateToolhelp32Snapshot.argtypes = [W.DWORD, W.DWORD]
k32.Process32FirstW.argtypes = [W.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
k32.Process32NextW.argtypes = [W.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
k32.OpenProcess.restype = W.HANDLE
k32.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
k32.CloseHandle.argtypes = [W.HANDLE]
k32.GetProcessTimes.argtypes = [W.HANDLE] + [ctypes.POINTER(W.FILETIME)] * 4
k32.K32GetProcessMemoryInfo.argtypes = [W.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), W.DWORD]
k32.GetProcessIoCounters.argtypes = [W.HANDLE, ctypes.POINTER(IO_COUNTERS)]
k32.GetSystemTimes.argtypes = [ctypes.POINTER(W.FILETIME)] * 3
k32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
ntdll.NtQueryInformationProcess.argtypes = [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.ULONG,
                                            ctypes.POINTER(W.ULONG)]
ntdll.NtQueryInformationProcess.restype = ctypes.c_long
iphlp.GetExtendedTcpTable.argtypes = [ctypes.c_void_p, ctypes.POINTER(W.DWORD), W.BOOL, W.ULONG,
                                      ctypes.c_int, W.ULONG]
iphlp.GetExtendedUdpTable.argtypes = iphlp.GetExtendedTcpTable.argtypes


def _ft(ft: W.FILETIME) -> int:
    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime


def _cmdline(h) -> str:
    size = W.ULONG(0)
    buf = ctypes.create_string_buffer(512)
    for _ in range(2):
        st = ntdll.NtQueryInformationProcess(h, 60, buf, len(buf), ctypes.byref(size))
        if st == 0:
            length = struct.unpack_from("<H", buf.raw, 0)[0]
            off = ctypes.sizeof(ctypes.c_void_p) * 2  # UNICODE_STRING header, then the text follows
            return buf.raw[off:off + length].decode("utf-16-le", "replace")
        if size.value > len(buf):
            buf = ctypes.create_string_buffer(size.value)
        else:
            return ""
    return ""


class WindowsBackend:
    name = "windows"

    def __init__(self):
        self._cmd_cache: dict[tuple[int, float], str] = {}

    def processes(self) -> dict[int, RawProc]:
        snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if snap == INVALID_HANDLE:
            raise ctypes.WinError(ctypes.get_last_error())
        out: dict[int, RawProc] = {}
        e = PROCESSENTRY32W()
        e.dwSize = ctypes.sizeof(e)
        try:
            ok = k32.Process32FirstW(snap, ctypes.byref(e))
            while ok:
                pid = e.th32ProcessID
                out[pid] = RawProc(pid, e.th32ParentProcessID, e.szExeFile, threads=e.cntThreads)
                ok = k32.Process32NextW(snap, ctypes.byref(e))
        finally:
            k32.CloseHandle(snap)
        c, x, kt, ut = W.FILETIME(), W.FILETIME(), W.FILETIME(), W.FILETIME()
        mem = PROCESS_MEMORY_COUNTERS()
        mem.cb = ctypes.sizeof(mem)
        io = IO_COUNTERS()
        alive = set()
        for pid, r in out.items():
            if pid == 0:
                continue
            h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not h:
                continue  # protected/system process: name + ppid only
            try:
                if k32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(x), ctypes.byref(kt), ctypes.byref(ut)):
                    r.cpu_time = (_ft(kt) + _ft(ut)) / 1e7
                    r.start_time = _ft(c) / 1e7
                if k32.K32GetProcessMemoryInfo(h, ctypes.byref(mem), mem.cb):
                    r.rss = mem.WorkingSetSize
                if k32.GetProcessIoCounters(h, ctypes.byref(io)):
                    r.io_read, r.io_write = io.ReadTransferCount, io.WriteTransferCount
                key = (pid, r.start_time)
                alive.add(key)
                if key not in self._cmd_cache:
                    self._cmd_cache[key] = _cmdline(h)
                r.cmdline = self._cmd_cache[key]
            finally:
                k32.CloseHandle(h)
        self._cmd_cache = {k: v for k, v in self._cmd_cache.items() if k in alive}
        return out

    def system(self) -> tuple[SystemStats, tuple[int, int]]:
        idle, kern, user = W.FILETIME(), W.FILETIME(), W.FILETIME()
        k32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
        ms = MEMORYSTATUSEX()
        ms.dwLength = ctypes.sizeof(ms)
        k32.GlobalMemoryStatusEx(ctypes.byref(ms))
        import os
        st = SystemStats(cores=os.cpu_count() or 1, mem_total=ms.ullTotalPhys,
                         mem_used=ms.ullTotalPhys - ms.ullAvailPhys)
        total = _ft(kern) + _ft(user)  # kernel time includes idle time on Windows
        return st, (total - _ft(idle), total)

    def listening(self) -> dict[int, list]:
        out: dict[int, list] = {}
        for fam, af in ((socket.AF_INET, 2), (socket.AF_INET6, 23)):
            for proto, fn, cls in (("tcp", iphlp.GetExtendedTcpTable, 3), ("udp", iphlp.GetExtendedUdpTable, 1)):
                size = W.DWORD(0)
                fn(None, ctypes.byref(size), False, af, cls, 0)
                if not size.value:
                    continue
                buf = ctypes.create_string_buffer(size.value)
                if fn(buf, ctypes.byref(size), False, af, cls, 0) != 0:
                    continue
                raw = buf.raw
                n = struct.unpack_from("<I", raw, 0)[0]
                for row in _rows(raw, n, proto, af):
                    out.setdefault(row[3], []).append(row[:3])
        for v in out.values():
            v.sort(key=lambda t: (t[2], t[0]))
        return out


def _port(dw: int) -> int:
    return ((dw & 0xFF) << 8) | ((dw >> 8) & 0xFF)


def _rows(raw: bytes, n: int, proto: str, af: int):
    """Yields (proto, addr, port, pid) for listening sockets."""
    if af == 2 and proto == "tcp":    # MIB_TCPROW_OWNER_PID: 6 DWORDs
        for i in range(n):
            st, la, lp, _, _, pid = struct.unpack_from("<6I", raw, 4 + i * 24)
            yield proto, socket.inet_ntoa(struct.pack("<I", la)), _port(lp), pid
    elif af == 2:                     # MIB_UDPROW_OWNER_PID: 3 DWORDs
        for i in range(n):
            la, lp, pid = struct.unpack_from("<3I", raw, 4 + i * 12)
            yield "udp", socket.inet_ntoa(struct.pack("<I", la)), _port(lp), pid
    elif proto == "tcp":              # MIB_TCP6ROW_OWNER_PID: 16+4+4+16+4+4+4+4 = 56 bytes
        for i in range(n):
            o = 4 + i * 56
            la = raw[o:o + 16]
            lp = struct.unpack_from("<I", raw, o + 20)[0]
            pid = struct.unpack_from("<I", raw, o + 52)[0]
            yield proto, "[" + socket.inet_ntop(socket.AF_INET6, la) + "]", _port(lp), pid
    else:                             # MIB_UDP6ROW_OWNER_PID: 16+4+4+4 = 28 bytes
        for i in range(n):
            o = 4 + i * 28
            la = raw[o:o + 16]
            lp, pid = struct.unpack_from("<2I", raw, o + 20)
            yield "udp", "[" + socket.inet_ntop(socket.AF_INET6, la) + "]", _port(lp), pid

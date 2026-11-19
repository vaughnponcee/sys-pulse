import io
import json
import os
import socket
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from syspulse import probes
from syspulse.backend_linux import LinuxBackend
from syspulse.model import Proc, RawProc, SystemStats, compute, parse_filter, subtree, tree_order
from syspulse.tui import ANSI, State, fit, handle_key, human, render, select_rows


def P(pid, ppid=0, name="x", cpu=0.0, rss=0, cmd="", tags=None, ports=None, gpu=0, rd=0.0, wr=0.0, thr=1):
    return Proc(pid, ppid, name, cpu, rss, rd, wr, thr, cmd, ports or [], gpu, tags or [])


class ComputeTests(unittest.TestCase):
    def test_rates(self):
        a = {1: RawProc(1, 0, "a", cpu_time=10.0, io_read=1000, io_write=0, start_time=5)}
        b = {1: RawProc(1, 0, "a", cpu_time=10.5, io_read=3000, io_write=500, start_time=5)}
        (p,) = compute(a, b, 0.5)
        self.assertAlmostEqual(p.cpu, 100.0)       # 0.5 s of CPU in 0.5 s = one full core
        self.assertAlmostEqual(p.read_bps, 4000.0)
        self.assertAlmostEqual(p.write_bps, 1000.0)

    def test_first_sample_and_pid_reuse(self):
        b = {1: RawProc(1, 0, "a", cpu_time=99.0, start_time=5)}
        self.assertEqual(compute(None, b, 1.0)[0].cpu, 0.0)
        a = {1: RawProc(1, 0, "old", cpu_time=50.0, start_time=1)}
        self.assertEqual(compute(a, b, 1.0)[0].cpu, 0.0)  # different start time: new process

    def test_counter_never_negative(self):
        a = {1: RawProc(1, 0, "a", cpu_time=5.0, io_read=100, start_time=1)}
        b = {1: RawProc(1, 0, "a", cpu_time=4.0, io_read=50, start_time=1)}
        p = compute(a, b, 1.0)[0]
        self.assertEqual((p.cpu, p.read_bps), (0.0, 0.0))


class TreeTests(unittest.TestCase):
    def test_order_and_depth(self):
        procs = [P(1), P(10, 1), P(11, 1), P(100, 10), P(50, 999)]  # 50's parent is gone
        out = tree_order(procs)
        self.assertEqual([(p.pid, p.depth) for p in out], [(1, 0), (10, 1), (100, 2), (11, 1), (50, 0)])

    def test_cycles_and_self_parent(self):
        procs = [P(1, 2), P(2, 1), P(3, 3), P(4, 3)]
        out = tree_order(procs)
        self.assertEqual(sorted(p.pid for p in out), [1, 2, 3, 4])  # nothing lost, no infinite loop

    def test_subtree(self):
        procs = [P(1), P(2, 1), P(3, 2), P(4, 1), P(5)]
        self.assertEqual(sorted(p.pid for p in subtree(procs, 2)), [2, 3])


class FilterTests(unittest.TestCase):
    procs = [P(1, name="python.exe", cpu=12, rss=600 * 2**20, cmd="python -m vllm.entrypoints",
               tags=["python", "ai:vllm"], ports=[("tcp", "0.0.0.0", 8000)]),
             P(2, name="svchost.exe", cpu=0.5, rss=20 * 2**20),
             P(3, name="ollama.exe", cpu=1, rss=2 * 2**30, tags=["ai:ollama", "gpu"], gpu=4 * 2**30,
               ports=[("tcp", "127.0.0.1", 11434)], rd=2e6)]

    def pids(self, expr):
        f = parse_filter(expr)
        return [p.pid for p in self.procs if f(p)]

    def test_terms(self):
        self.assertEqual(self.pids("python"), [1])
        self.assertEqual(self.pids("vllm"), [1])            # matches command line
        self.assertEqual(self.pids("!svchost"), [1, 3])
        self.assertEqual(self.pids("cpu>5"), [1])
        self.assertEqual(self.pids("mem>500M"), [1, 3])
        self.assertEqual(self.pids("mem>=2G"), [3])
        self.assertEqual(self.pids("gpu>0"), [3])
        self.assertEqual(self.pids("io>1M"), [3])
        self.assertEqual(self.pids("port:11434"), [3])
        self.assertEqual(self.pids("pid=2"), [2])
        self.assertEqual(self.pids("ai"), [1, 3])
        self.assertEqual(self.pids("listen !ollama"), [1])

    def test_bad_term(self):
        with self.assertRaises(ValueError):
            parse_filter("mem>lots")


class ProbeTests(unittest.TestCase):
    def test_nvidia_parsing(self):
        g = probes.parse_gpu_csv("NVIDIA GeForce RTX 2070, 45, 397, 8192, 61\nTesla T4, [N/A], 100, 15360, [N/A]\n")
        self.assertEqual(g[0]["util"], 45.0)
        self.assertEqual(g[0]["mem_total"], 8192 * 2**20)
        self.assertIsNone(g[1]["util"])
        apps = probes.parse_gpu_apps("1234, 5120\n1632, [N/A]\n1234, 1024\n")
        self.assertEqual(apps, {1234: 6144 * 2**20, 1632: 0})

    def test_ai_tagging(self):
        procs = [P(1, name="ollama.exe"), P(2, name="python.exe", cmd="python main.py --listen --port 8188"),
                 P(3, name="python3", cmd="python app.py"), P(4, name="python.exe", cmd="python server.py"),
                 P(5, name="dwm.exe"), P(6, name="worker", cmd="./train")]
        ports = {4: [("tcp", "0.0.0.0", 7860)]}
        probes.tag(procs, ports, {5: 0, 6: 3 * 2**30})
        tags = {p.pid: p.tags for p in procs}
        self.assertIn("ai:ollama", tags[1])
        self.assertIn("ai:comfyui", tags[2])
        self.assertEqual(tags[3], ["python"])
        self.assertIn("ai:gradio", tags[4])                 # port hint for a python process
        self.assertEqual(tags[5], ["gpu"])                   # uses GPU, unknown memory: not AI
        self.assertEqual(tags[6], ["gpu", "ai:gpu-compute"])  # big allocation


class LinuxBackendTests(unittest.TestCase):
    def test_parse_stat_with_hostile_comm(self):
        line = "4242 (my (weird) proc) S 1 4242 4242 0 -1 4194560 100 0 0 0 250 50 0 0 20 0 7 0 12345 0 5000 18446744073709551615"
        s = LinuxBackend.parse_stat(line)
        self.assertEqual((s["comm"], s["ppid"], s["utime"], s["stime"], s["threads"], s["starttime"], s["rss_pages"]),
                         ("my (weird) proc", 1, 250, 50, 7, 12345, 5000))

    def test_parse_net(self):
        tcp = ("  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
               "   0: 0100007F:2CAA 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000 0 31337 1\n"
               "   1: 0100007F:2CAB 0100007F:9C40 01 00000000:00000000 00:00000000 00000000  1000 0 4444 1\n")
        self.assertEqual(LinuxBackend.parse_net(tcp, "tcp", False), [("tcp", "127.0.0.1", 11434, 31337)])
        tcp6 = ("  sl  local_address                         remote_address                        st ...\n"
                "   0: 00000000000000000000000001000000:1F90 00000000000000000000000000000000:0000 0A "
                "00000000:00000000 00:00000000 00000000  1000 0 999 1\n")
        self.assertEqual(LinuxBackend.parse_net(tcp6, "tcp", True), [("tcp", "[::1]", 8080, 999)])

    def test_fixture_proc_tree(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "stat").write_text("cpu  100 0 50 800 50 0 0 0 0 0\ncpu0 1 1 1 1\n")
            (root / "meminfo").write_text("MemTotal:       16000000 kB\nMemFree:  1000 kB\nMemAvailable:    4000000 kB\n")
            for pid, ppid, comm in ((1, 0, "init"), (77, 1, "python3")):
                pd = root / str(pid)
                pd.mkdir()
                (pd / "stat").write_text(f"{pid} ({comm}) S {ppid} 0 0 0 -1 0 0 0 0 0 300 100 0 0 20 0 3 0 500 0 256 0")
                (pd / "cmdline").write_bytes(b"python3\0-m\0http.server\0")
                (pd / "io").write_text("rchar: 1\nwchar: 2\nread_bytes: 4096\nwrite_bytes: 8192\n")
            b = LinuxBackend(d)
            b.hz, b.page = 100, 4096
            procs = b.processes()
            self.assertEqual(procs[77].ppid, 1)
            self.assertEqual(procs[77].cpu_time, 4.0)
            self.assertEqual(procs[77].rss, 256 * 4096)
            self.assertEqual(procs[77].io_write, 8192)
            self.assertEqual(procs[77].cmdline, "python3 -m http.server")
            st, (busy, total) = b.system()
            self.assertEqual((busy, total), (150, 1000))
            self.assertEqual(st.mem_used, 12000000 * 1024)


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.st = SystemStats(cpu_percent=42.0, cores=8, mem_total=16 * 2**30, mem_used=8 * 2**30,
                              gpus=[{"name": "RTX", "util": 50.0, "mem_used": 2**30, "mem_total": 8 * 2**30, "temp": 60}])
        self.procs = [P(1, name="init", cpu=0.1), P(2, 1, "python.exe", 75.0, 2**30, "python -m vllm", ["ai:vllm"],
                                                     [("tcp", "0.0.0.0", 8000), ("tcp", "[::]", 8000)]),
                      P(3, 1, "bash", 2.0, 2**20)]

    def frame(self, state, color=False, w=100, h=20):
        return [ANSI.sub("", ln) for ln in render(state, self.procs, self.st, w, h, color)]

    def test_shape_and_widths(self):
        for color in (False, True):
            out = render(State(), self.procs, self.st, 90, 15, color)
            self.assertEqual(len(out), 15)
            self.assertTrue(all(len(ANSI.sub("", ln)) == 90 for ln in out))

    def test_sorted_by_cpu_with_header(self):
        f = self.frame(State())
        self.assertIn("CPU [", f[0])
        self.assertIn("42.0%", f[0])
        self.assertIn("GPU0", f[2])
        rows = [ln for ln in f if "python.exe" in ln or "bash" in ln or "init" in ln]
        self.assertTrue(rows[0].lstrip("> ").startswith("2"))  # highest CPU first

    def test_ports_view_dedupes_v4_v6(self):
        f = self.frame(State(view="ports"))
        row = next(ln for ln in f if "python.exe" in ln)
        self.assertIn("8000", row)
        self.assertIn("+1", row)            # two addresses, one port
        self.assertNotIn("also", row)

    def test_tree_and_filter_error(self):
        f = self.frame(State(tree=True, sort="pid"))
        self.assertTrue(any("└ python.exe" in ln for ln in f))
        f = self.frame(State(filter="mem>lots"))
        self.assertIn("filter error", f[-1])

    def test_fit_ansi_aware(self):
        s = "\033[31mhello\033[0m world"
        self.assertEqual(ANSI.sub("", fit(s, 8)), "hello wo")
        self.assertEqual(len(ANSI.sub("", fit("ab", 5))), 5)

    def test_human(self):
        self.assertEqual(human(512), "512B")
        self.assertEqual(human(1536), "1.5KB")
        self.assertEqual(human(3 * 2**30), "3.0GB")


class KeyTests(unittest.TestCase):
    def test_vim_navigation_and_modes(self):
        st = State()
        procs = [P(i, 0, f"p{i}", cpu=float(i)) for i in range(1, 30)]
        render(st, procs, SystemStats(), 80, 15, False)
        handle_key(st, "j", 10)
        handle_key(st, "j", 10)
        self.assertEqual(st.sel, 2)
        handle_key(st, "G", 10)
        self.assertEqual(st.sel, 28)
        handle_key(st, "g", 10)
        self.assertEqual(st.sel, 0)
        for k in "/p1":
            handle_key(st, k, 10)
        self.assertTrue(st.editing)
        handle_key(st, "ENTER", 10)
        self.assertEqual(st.filter, "p1")
        handle_key(st, "s", 10)
        self.assertEqual(st.sort, "mem")
        handle_key(st, "t", 10)
        self.assertTrue(st.tree)
        render(st, procs, SystemStats(), 80, 15, False)
        handle_key(st, "ENTER", 10)
        self.assertIsNotNone(st.focus)
        handle_key(st, "BACKSPACE", 10)
        self.assertIsNone(st.focus)
        self.assertFalse(handle_key(st, "q", 10))

    def test_ai_only_selects_tagged(self):
        st = State(ai_only=True)
        rows = select_rows(st, [P(1, tags=["gpu"]), P(2, tags=["ai:ollama"])])
        self.assertEqual([p.pid for p in rows], [2])


@unittest.skipUnless(sys.platform == "win32" or sys.platform.startswith("linux"), "live backend only on Win/Linux")
class LiveBackendTests(unittest.TestCase):
    def test_sees_itself_and_its_listening_socket(self):
        from syspulse.collector import Collector
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen()
        port = srv.getsockname()[1]
        try:
            c = Collector(gpu=probes.Gpu(every=1e9))
            c.sample()
            sum(i * i for i in range(2_000_000))  # burn a little CPU between samples
            time.sleep(0.1)
            procs, st = c.sample()
        finally:
            srv.close()
        me = next(p for p in procs if p.pid == os.getpid())
        self.assertGreater(me.rss, 5 * 2**20)
        self.assertIn("python", (me.name + me.cmdline).lower())
        self.assertIn(port, [pt for _, _, pt in me.ports])
        self.assertGreater(me.cpu, 1.0)
        self.assertGreater(st.mem_total, 0)
        self.assertGreater(len(procs), 10)

    def test_cli_json(self):
        from syspulse.cli import main
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(main(["--once", "--json", "-n", "5", "-i", "0.2"]), 0)
        data = json.loads(buf.getvalue())
        self.assertEqual(len(data["processes"]), 5)
        self.assertIn("cpu_percent", data)


if __name__ == "__main__":
    unittest.main()

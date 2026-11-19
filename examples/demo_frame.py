"""Render sample frames with the real renderer and synthetic data (used for
the README, so no real machine's process list gets published)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syspulse.model import Proc, SystemStats  # noqa: E402
from syspulse.tui import State, render  # noqa: E402

G = 2**30
M = 2**20
procs = [
    Proc(4120, 3880, "ollama.exe", 182.4, 5.6 * G, 0, 1.2 * M, 38, "ollama serve",
         [("tcp", "127.0.0.1", 11434)], 4.9 * G, ["ai:ollama", "gpu"]),
    Proc(6232, 5108, "python.exe", 96.0, 3.1 * G, 18 * M, 64 * 1024, 41,
         "python main.py --listen --port 8188", [("tcp", "0.0.0.0", 8188)], 2.2 * G,
         ["python", "ai:comfyui", "gpu"]),
    Proc(7710, 5108, "python.exe", 12.5, 412 * M, 0, 0, 9, "python -m jupyter lab",
         [("tcp", "127.0.0.1", 8888)], 0, ["python", "ai:jupyter"]),
    Proc(2210, 880, "Code.exe", 6.1, 610 * M, 120 * 1024, 40 * 1024, 52, "Code.exe --type=renderer", [], 0, []),
    Proc(880, 4, "explorer.exe", 1.2, 150 * M, 0, 0, 90, r"C:\Windows\Explorer.EXE", [], 0, []),
    Proc(3880, 880, "WindowsTerminal.exe", 0.8, 95 * M, 0, 0, 22, "", [], 0, []),
    Proc(5108, 3880, "pwsh.exe", 0.3, 88 * M, 0, 0, 14, "", [], 0, []),
    Proc(4, 0, "System", 0.4, 0, 0, 2.5 * M, 240, "", [("tcp", "0.0.0.0", 445)], 0, []),
]
st = SystemStats(cpu_percent=31.7, cores=12, mem_total=32 * G, mem_used=14.2 * G,
                 gpus=[{"name": "NVIDIA GeForce RTX 4070", "util": 87.0, "mem_used": 7.4 * G,
                        "mem_total": 12 * G, "temp": 71}])
for title, s in (("default (sort cpu)", State()), ("/ai, tree", State(ai_only=True, tree=True, sort="pid")),
                 ("ports (p)", State(view="ports"))):
    print(f"--- {title}")
    for line in render(s, procs, st, 118, 16, False, 41, "windows"):
        if line.strip():
            print(line.rstrip())

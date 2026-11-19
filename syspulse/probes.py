"""GPU (nvidia-smi) and AI-runtime detection."""
from __future__ import annotations

import re
import shutil
import subprocess

from .model import Proc


def parse_gpu_csv(text: str) -> list[dict]:
    """nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu
    --format=csv,noheader,nounits"""
    out = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue

        def num(s):
            try:
                return float(s)
            except ValueError:
                return None  # "[N/A]" on some laptop GPUs

        out.append({"name": parts[0], "util": num(parts[1]), "mem_used": (num(parts[2]) or 0) * 2**20,
                    "mem_total": (num(parts[3]) or 0) * 2**20, "temp": num(parts[4])})
    return out


def parse_gpu_apps(text: str) -> dict[int, int]:
    """nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits"""
    out: dict[int, int] = {}
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                out[int(parts[0])] = out.get(int(parts[0]), 0) + int(float(parts[1]) * 2**20)
            except ValueError:
                out[int(parts[0])] = out.get(int(parts[0]), 0)
    return out


class Gpu:
    """Polls nvidia-smi at most every `every` seconds (it costs ~50-150 ms)."""

    def __init__(self, every: float = 3.0):
        self.exe = shutil.which("nvidia-smi")
        self.every = every
        self._last = -1e9
        self.gpus: list[dict] = []
        self.apps: dict[int, int] = {}

    def poll(self, now: float):
        if not self.exe or now - self._last < self.every:
            return
        self._last = now
        try:
            g = subprocess.run([self.exe, "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
                                "temperature.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            a = subprocess.run([self.exe, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            self.gpus = parse_gpu_csv(g.stdout)
            self.apps = parse_gpu_apps(a.stdout)
        except (OSError, subprocess.TimeoutExpired):
            self.exe = None  # broken driver: stop trying


# ---------------------------------------------------------------- AI runtimes
NAME_RULES = [
    (r"^ollama", "ollama"), (r"^llama-(server|cli)|^llama\.cpp|^llamafile", "llama.cpp"),
    (r"^lm ?studio|^lms(\.exe)?$", "lmstudio"), (r"^koboldcpp", "koboldcpp"),
    (r"^text-generation-(launcher|router)", "tgi"), (r"^jan(\.exe)?$", "jan"),
    (r"^gpt4all", "gpt4all"), (r"^local-?ai", "localai"),
]
CMD_RULES = [
    (r"\bvllm\b", "vllm"), (r"llama_cpp|llama-cpp-python", "llama.cpp"), (r"\bollama\b", "ollama"),
    (r"comfyui|comfy/main\.py|main\.py .*--listen", "comfyui"),
    (r"stable-diffusion-webui|automatic1111|webui\.py|launch\.py", "sd-webui"),
    (r"text-generation-webui|oobabooga", "textgen-webui"), (r"\bmlx_lm\b", "mlx-lm"),
    (r"exllama", "exllama"), (r"whisper", "whisper"), (r"\btorch\b|transformers|accelerate launch", "torch"),
    (r"\bgradio\b", "gradio"), (r"jupyter|ipykernel", "jupyter"),
]
GPU_AI_MIN_BYTES = 512 * 2**20
PORT_HINTS = {11434: "ollama", 1234: "lmstudio", 8080: "llama.cpp?", 7860: "gradio", 8188: "comfyui",
              5001: "koboldcpp", 8888: "jupyter", 8000: "api?"}


def tag(procs: list[Proc], ports: dict[int, list], gpu_apps: dict[int, int]) -> None:
    for p in procs:
        name = p.name.lower()
        cmd = p.cmdline.lower()
        tags = []
        if re.match(r"^python(w|3(\.\d+)?)?(\.exe)?$", name) or name in ("py.exe", "uv.exe", "uvicorn"):
            tags.append("python")
        for pat, label in NAME_RULES:
            if re.search(pat, name):
                tags.append(f"ai:{label}")
                break
        else:
            for pat, label in CMD_RULES:
                if re.search(pat, cmd):
                    tags.append(f"ai:{label}")
                    break
        p.ports = ports.get(p.pid, [])
        if not any(t.startswith("ai:") for t in tags) and "python" in tags:
            for _, _, port in p.ports:
                if port in PORT_HINTS:
                    tags.append(f"ai:{PORT_HINTS[port]}")
                    break
        if p.pid in gpu_apps:
            p.gpu_mem = gpu_apps[p.pid]
            tags.append("gpu")
            # Under Windows WDDM, nvidia-smi lists every process that touches the
            # GPU (dwm.exe, browsers) with memory "[N/A]". Only a large, known
            # allocation is evidence of an ML workload.
            if p.gpu_mem >= GPU_AI_MIN_BYTES and not any(t.startswith("ai:") for t in tags):
                tags.append("ai:gpu-compute")
        p.tags = tags

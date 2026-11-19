# sys-pulse

sys-pulse is a low-overhead terminal dashboard for **Windows and Linux**. It shows:

- per-process CPU, memory, disk I/O rates, threads and command lines
- listening TCP/UDP ports per process
- NVIDIA GPU utilization, plus GPU memory per process
- tags on local AI/ML runtimes: Ollama, llama.cpp, LM Studio, vLLM, ComfyUI, Jupyter, Gradio, and Python processes in general

It has vim-style navigation and filtering and a process-tree view. It needs only the Python 3.10+ standard library: `ctypes` calls into Win32 on Windows, and it reads `/proc` on Linux.

```bash
python -m syspulse                         # interactive TUI
python -m syspulse --once -s mem -n 20     # one snapshot table, sorted by memory
python -m syspulse --once --ports          # who is listening on what
python -m syspulse --once --ai --json      # AI runtimes as JSON (for scripts)
python -m syspulse --once -t -f "python cpu>1"
python -m unittest discover -s tests -t .  # 23 tests
```

## Screens

These frames are rendered by `python examples/demo_frame.py` with the real renderer and synthetic data. The live tool looks the same, with color.

```
 CPU [||||||||||||                            ]  31.7%  12 cores   8 procs  sample 41 ms  windows
 MEM [|||||||||||||||||                       ] 14.2GB / 32.0GB
 GPU0 [||||||||||||||||||||||||||||||||||      ]  87.0%  7.4GB / 12.0GB 71C NVIDIA GeForce RTX 4070
 procs  sort:cpu
    PID    PPID   CPU%      MEM   READ/s  WRITE/s     GPU  THR  NAME / COMMAND
>  4120    3880  182.4    5.6GB       0B    1.2MB   4.9GB   38  ollama.exe [ai:ollama gpu]  ollama serve
   6232    5108   96.0    3.1GB   18.0MB   64.0KB   2.2GB   41  python.exe [ai:comfyui gpu]  python main.py --listen -
   7710    5108   12.5  412.0MB       0B       0B       -    9  python.exe [ai:jupyter]  python -m jupyter lab
   2210     880    6.1  610.0MB  120.0KB   40.0KB       -   52  Code.exe  Code.exe --type=renderer
    880       4    1.2  150.0MB       0B       0B       -   90  explorer.exe  C:\Windows\Explorer.EXE
```

```
 ports  sort:cpu
    PID NAME                   PROTO ADDRESS                        PORT  TAGS
>     4 System                 tcp   0.0.0.0                         445
   6232 python.exe             tcp   0.0.0.0                        8188  ai:comfyui gpu
   7710 python.exe             tcp   127.0.0.1                      8888  ai:jupyter
   4120 ollama.exe             tcp   127.0.0.1                     11434  ai:ollama gpu
```

## Keys and filters

| key | action | key | action |
|---|---|---|---|
| `j` `k` / arrows | move | `g` `G` | top / bottom |
| PgUp PgDn | page | `s` | cycle sort: cpu, mem, io, gpu, pid, name |
| `/` | edit filter | Esc | clear filter |
| `t` | tree view | `p` | ports view |
| `a` | AI runtimes only | Enter | focus the selected process's subtree |
| Backspace | leave focus | space | pause |
| `?` | help | `q` | quit |

Filter terms are separated by spaces, and every term must match:

| term | matches |
|---|---|
| `python` | the text appears in the name or command line (case-insensitive) |
| `!svchost` | negation |
| `cpu>5`, `mem>500M`, `io>1M`, `gpu>0`, `threads>50` | numeric comparisons; sizes take K, M or G |
| `pid=1234`, `ppid:4`, `port:8080` | exact matches |
| `ai`, `listen` | tagged AI runtime / has a listening port |

`--once` accepts the same filter as `-f "..."`.

## How it collects

| data | Windows (`ctypes`) | Linux |
|---|---|---|
| process list, parent, threads | `CreateToolhelp32Snapshot` + `Process32NextW` | `/proc/<pid>/stat` (split on the last `)`, because `comm` can contain `)`) |
| CPU time, start time | `GetProcessTimes` (100 ns FILETIMEs) | `utime + stime`, `starttime` / `SC_CLK_TCK` |
| memory | `K32GetProcessMemoryInfo` → working set | `rss` pages × `SC_PAGE_SIZE` |
| disk I/O | `GetProcessIoCounters` | `/proc/<pid>/io` `read_bytes` / `write_bytes` |
| command line | `NtQueryInformationProcess(ProcessCommandLineInformation)`, cached per (pid, start time) | `/proc/<pid>/cmdline` |
| listening ports | `GetExtendedTcpTable` / `GetExtendedUdpTable` (IPv4 and IPv6, owner PID) | `/proc/net/{tcp,udp}{,6}` socket inodes joined to `/proc/<pid>/fd/*` |
| system CPU and memory | `GetSystemTimes`, `GlobalMemoryStatusEx` | `/proc/stat`, `/proc/meminfo` |
| GPU | `nvidia-smi --query-gpu` and `--query-compute-apps`, at most every 3 s | same |

How the numbers are computed:
- **Rates come from two samples.** For example, CPU% = Δ(user + kernel time) / Δwall × 100, top-style, so 182% means about 1.8 cores.
- **Reused PIDs are handled.** A PID whose start time changed between samples is treated as a new process, so its rate starts at 0 instead of a large wrong delta.
- **Ports are polled less often.** Socket tables are read every 5 s, because they change slowly and the Linux inode join is the most expensive step.
- **Restricted processes stay listed.** Protected or other users' processes that can't be opened still appear with their name and parent; their counters show 0.

**AI tagging** combines three signals:
- a process name (for example `ollama`, `llama-server`, `LM Studio`, `koboldcpp`);
- a command-line pattern (`vllm`, `llama_cpp`, ComfyUI's `main.py --listen`, `stable-diffusion-webui`, `jupyter`, `gradio`, `torch`/`transformers`);
- for Python processes only, a well-known port (11434, 1234, 7860, 8188, 8888, ...).

GPU use alone is **not** treated as AI. On Windows, `nvidia-smi` lists every process that touches the GPU, including `dwm.exe` and browsers, with memory `[N/A]`. The first version tagged all of them as `ai:gpu-compute`. Now a process gets that tag only with at least 512 MiB of reported GPU memory.

## Overhead

Measured on Windows 11, 12 logical cores, about 308 processes, Python 3.14:

| measure | value |
|---|---|
| one full sample (processes, system, ports) | 38–46 ms, median 41 ms |
| sys-pulse's own CPU at the default 1.5 s refresh | 3.2% of one core (0.27% of the machine) |

Most of the sample time goes to opening a handle to each process. Command lines are cached per (pid, start time), so they are read once per process lifetime.

## Tests

`tests/test_syspulse.py` has 23 tests:
- **Rate maths:** CPU, I/O, reused PIDs, counters that run backwards.
- **Tree building:** orphaned processes, parent cycles, processes that list themselves as parent.
- **Filters:** every filter term.
- **Parsers:** `nvidia-smi` CSV (including `[N/A]`), and Linux `/proc` against a fixture tree, including a `comm` containing `(weird)` characters and IPv6 socket rows.
- **AI tagging rules.**
- **The renderer:** exact frame width with ANSI codes, IPv4/IPv6 duplicates collapsed in the ports view, tree drawing.
- **Vim key handling.**
- **A live test:** it opens a listening socket and burns some CPU, then checks that the live backend reports its own PID with that port, more than 5 MB RSS, more than 1% CPU, and a Python command line.

The Windows backend was tested live. The Linux backend is covered by fixture tests only; it has not been run on a real Linux machine here.

## Limitations

- **No macOS backend.** macOS would need `libproc` through ctypes.
- **GPU support is NVIDIA-only,** through `nvidia-smi`. AMD (`rocm-smi`) and Intel are not supported.
- **Per-process GPU memory is unavailable on Windows WDDM,** where `nvidia-smi` reports `[N/A]`. Only Linux and TCC-mode GPUs give per-process numbers.
- **Memory is the working set or RSS,** not private bytes or proportional set size, so shared pages are counted in every process that maps them.
- **No kill or renice.** It is a monitor; nothing in it changes processes.
- **A ports-only view can be up to 5 s stale** (`ports_every`).

## License

MIT

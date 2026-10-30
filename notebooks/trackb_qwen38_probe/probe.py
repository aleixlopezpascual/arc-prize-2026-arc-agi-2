# Track B go/no-go probe: can ONE L4 serve Qwen3.8-27B (4-bit GPTQ) with vLLM,
# installed into an isolated dir so the image's torch/Unsloth (used by NVARC) is untouched?
# Writes /kaggle/working/probe.json with boot result, VRAM, and aggregate decode tok/s.
import glob, json, os, subprocess, sys, threading, time, urllib.request

T0 = time.time()
HARD_DEADLINE = T0 + 100 * 60  # whole probe must finish in 100 min
OUT = "/kaggle/working/probe.json"
report = {"stages": {}}


def save():
    report["elapsed_s"] = round(time.time() - T0, 1)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)


def sh(cmd, timeout=600):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)[-3000:]


# ---- 1. Base image facts (what NVARC sees) ----
rc, smi = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
report["gpus"] = smi.strip().splitlines()
rc, base = sh(f"{sys.executable} -c \"import torch;print(torch.__version__, torch.version.cuda)\"")
report["base_torch_before"] = base.strip()
save()

# ---- 2. Locate inputs ----
wheel_dirs = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/requirements.lock", recursive=True)]
report["wheel_dirs"] = wheel_dirs
models = sorted({os.path.dirname(p) for p in glob.glob("/kaggle/input/**/config.json", recursive=True)
                 if "27b" in p.lower()})
report["model_dirs"] = models
save()
assert wheel_dirs, "vLLM wheelhouse not attached"
assert models, "no Qwen3.8-27B model attached"
WHEELS = wheel_dirs[0]

# ---- 3. Isolated install (never touch the system site-packages) ----
VENV = "/kaggle/tmp/vllm_env"
t = time.time()
rc, out = sh(f"{sys.executable} -m pip install --no-index --find-links {WHEELS} "
             f"--target {VENV} -r {WHEELS}/requirements.lock -q", timeout=1500)
report["stages"]["install"] = {"rc": rc, "s": round(time.time() - t, 1), "tail": out[-1500:]}
rc, base_after = sh(f"{sys.executable} -c \"import torch;print(torch.__version__, torch.version.cuda)\"")
report["base_torch_after"] = base_after.strip()  # must equal base_torch_before
save()

env = os.environ.copy()
env["PYTHONPATH"] = VENV + os.pathsep + env.get("PYTHONPATH", "")
real_lib = "/usr/local/nvidia/lib64"
if os.path.exists(real_lib + "/libcuda.so"):
    env["LD_LIBRARY_PATH"] = real_lib + ":" + env.get("LD_LIBRARY_PATH", "")
    env["LD_PRELOAD"] = real_lib + "/libcuda.so"
env["VLLM_FORCE_ATTN_BACKEND"] = "TRITON_ATTN"  # FlashInfer FP8-KV kernels are SM90+ only; L4 is SM89
env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
env["CUDA_VISIBLE_DEVICES"] = "0"

p = subprocess.run([sys.executable, "-c", "import torch,vllm;print(torch.__version__,torch.cuda.is_available(),vllm.__version__)"],
                   env=env, capture_output=True, text=True, timeout=300)
report["venv_torch_vllm"] = (p.stdout + p.stderr).strip()[-800:]
save()


# ---- 4. Boot vLLM on a single L4, try each model / config until one serves ----
def boot(model, max_len, spec):
    log = f"/kaggle/working/vllm_{len(report['stages'])}.log"
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
           "--served-model-name", "q", "--port", "8000", "--tensor-parallel-size", "1",
           "--gpu-memory-utilization", "0.95", "--max-model-len", str(max_len),
           "--max-num-seqs", "16", "--kv-cache-dtype", "fp8", "--attention-backend", "TRITON_ATTN",
           "--trust-remote-code", "--limit-mm-per-prompt", '{"image":0,"video":0}',
           "--reasoning-parser", "qwen3"]
    if spec:
        cmd += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 2})]
    fh = open(log, "w")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    t = time.time()
    while time.time() - t < 900 and time.time() < HARD_DEADLINE:
        if proc.poll() is not None:
            break
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3)
            return proc, log, round(time.time() - t, 1)
        except Exception:
            time.sleep(3)
    os.killpg(proc.pid, 9) if proc.poll() is None else None
    return None, log, round(time.time() - t, 1)


def log_facts(log):
    keep = [l.strip()[:220] for l in open(log, errors="ignore")
            if any(k in l for k in ("KV cache", "Maximum concurrency", "Model loading took", "GiB",
                                    "Error", "error", "CUDA out of memory", "attention backend"))]
    return keep[-25:]


server = None
for model in models:
    for max_len, spec in [(65536, True), (65536, False), (32768, False)]:
        proc, log, s = boot(model, max_len, spec)
        key = f"boot:{os.path.basename(os.path.dirname(model))}:{max_len}:spec={spec}"
        report["stages"][key] = {"ok": proc is not None, "s": s, "log": log_facts(log)}
        save()
        if proc:
            server = (proc, model, max_len, spec, log)
            break
    if server:
        break
assert server, "vLLM failed to boot on a single L4 with every model/config"
rc, mem = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader")
report["vram_used_after_boot"] = mem.strip().splitlines()

# ---- 5. Throughput: 16 concurrent code-writing requests on a real eval task ----
tasks = json.load(open(glob.glob("/kaggle/input/**/arc-agi_evaluation_challenges.json", recursive=True)[0]))
tid, task = next(iter(tasks.items()))
prompt = ("Write a Python function transform(grid: list[list[int]]) -> list[list[int]] that maps each "
          "input to its output for these ARC examples. Return only code.\n" +
          "\n".join(f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in task["train"]))
results = []


def one(i):
    body = json.dumps({"model": "q", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 3000, "temperature": 0.8, "seed": i}).encode()
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=1200))
        results.append({"tok": r["usage"]["completion_tokens"], "s": time.time() - t})
    except Exception as e:
        results.append({"err": repr(e)[:200]})


t = time.time()
threads = [threading.Thread(target=one, args=(i,)) for i in range(16)]
[th.start() for th in threads]
[th.join(timeout=max(1, HARD_DEADLINE - time.time())) for th in threads]
wall = time.time() - t
toks = sum(r.get("tok", 0) for r in results)
report["bench"] = {"model": server[1], "max_len": server[2], "mtp": server[3], "task": tid,
                   "requests": len(results), "errors": sum("err" in r for r in results),
                   "completion_tokens": toks, "wall_s": round(wall, 1),
                   "aggregate_tok_s_one_L4": round(toks / max(wall, 1e-9), 1),
                   "prompt_chars": len(prompt)}
report["bench"]["log"] = log_facts(server[4])
save()
os.killpg(server[0].pid, 9)
print(json.dumps(report, indent=2))

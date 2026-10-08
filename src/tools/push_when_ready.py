"""Pushes hybrid submission notebook as soon as a Kaggle GPU slot becomes available."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def push_when_ready(poll_sec: int = 20, max_wait_min: int = 60) -> bool:
    print("[*] Waiting for Kaggle batch GPU slot to free up...")
    t_start = time.time()
    nb_dir = str(ROOT / "notebooks" / "submissions" / "hybrid_sota_v1")

    while (time.time() - t_start) < max_wait_min * 60:
        res = subprocess.run(
            ["kaggle", "kernels", "push", "-p", nb_dir],
            capture_output=True,
            text=True,
        )
        out = (res.stdout + "\n" + res.stderr).strip()
        utc = time.strftime("%H:%M:%S UTC", time.gmtime())
        if "successfully pushed" in out:
            print(f"[{utc}] [+] SUCCESS! {out}")
            return True
        elif "Maximum batch GPU session count" in out:
            print(f"[{utc}] Waiting for Kaggle GPU slot (2/2 active)... retry in {poll_sec}s")
        else:
            print(f"[{utc}] Result: {out}... retry in {poll_sec}s")

        time.sleep(poll_sec)

    print("[!] Timed out waiting for Kaggle GPU slot.")
    return False


if __name__ == "__main__":
    push_when_ready()

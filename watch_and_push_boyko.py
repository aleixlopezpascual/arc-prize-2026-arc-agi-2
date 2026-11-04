#!/usr/bin/env python3
"""Watch for Kaggle GPU quota reset, push Boyko dual-pass fork, and monitor execution."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent
REPO_ROOT = Path("/Users/aleix.lopez/arc-prize-2026-arc-agi-2")
NB_DIR = WORKTREE / "notebooks" / "boyko_dualpass_fork"
KERNEL_SLUG = "aleixlopez/arc-agi2-boyko-dualpass-fork"
LOG_FILE = WORKTREE / "watch_boyko.log"


def log(msg: str):
    utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    formatted = f"[{utc}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(formatted + "\n")


def push_kernel(poll_sec: int = 20, max_wait_min: int = 60) -> bool:
    log(f"[*] Starting push polling loop for {KERNEL_SLUG}...")
    start_time = time.time()
    attempt = 0

    while time.time() - start_time < max_wait_min * 60:
        attempt += 1
        res = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(NB_DIR)],
            capture_output=True,
            text=True,
            cwd=str(WORKTREE),
        )
        combined = (res.stdout + "\n" + res.stderr).strip()

        if "successfully pushed" in combined.lower():
            log(f"[+] Push succeeded on attempt {attempt}!")
            log(combined)
            return True

        if "quota" in combined.lower():
            if attempt % 3 == 1:
                log(f"[*] Attempt {attempt}: GPU quota still exhausted. Retrying in {poll_sec}s...")
        elif "session count" in combined.lower():
            log(f"[*] Attempt {attempt}: GPU slots full (2/2 active). Retrying in {poll_sec}s...")
        else:
            log(f"[?] Attempt {attempt} result: {combined}. Retrying in {poll_sec}s...")

        time.sleep(poll_sec)

    log("[-] Timed out waiting for GPU quota reset.")
    return False


def monitor_kernel(poll_sec: int = 60, timeout_min: int = 150) -> bool:
    log(f"[*] Monitoring kernel execution: {KERNEL_SLUG}...")
    start_time = time.time()
    last_status = None

    while time.time() - start_time < timeout_min * 60:
        res = subprocess.run(
            ["kaggle", "kernels", "status", KERNEL_SLUG],
            capture_output=True,
            text=True,
        )
        out = (res.stdout + "\n" + res.stderr).strip()

        status = None
        for line in out.splitlines():
            if KERNEL_SLUG in line and "status" in line:
                status = line
                break
        if not status:
            status = out

        if status != last_status:
            log(f"[*] Status change: {status}")
            last_status = status

        if "COMPLETE" in status:
            log("[+] Kernel run completed successfully!")
            return True
        elif "ERROR" in status or "FAILED" in status:
            log(f"[-] Kernel run failed with status: {status}")
            return False

        time.sleep(poll_sec)

    log("[-] Kernel execution monitoring timed out.")
    return False


def verify_outputs():
    out_dir = WORKTREE / "out_boyko"
    out_dir.mkdir(exist_ok=True)
    log(f"[*] Downloading kernel output to {out_dir}...")
    res = subprocess.run(
        ["kaggle", "kernels", "output", KERNEL_SLUG, "-p", str(out_dir)],
        capture_output=True,
        text=True,
    )
    log(res.stdout + "\n" + res.stderr)

    sub_file = out_dir / "submission.json"
    if not sub_file.exists():
        log(f"[-] {sub_file} does not exist!")
        return False

    chal_file = REPO_ROOT / "input" / "arc-agi_evaluation_challenges.json"
    if not chal_file.exists():
        chal_file = REPO_ROOT / "input" / "arc-agi_test_challenges.json"

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from validator import validate_submission_file

    valid, errs = validate_submission_file(sub_file, chal_file)
    if valid:
        log(f"[+] Output validation PASSED against {chal_file.name}!")
        return True
    else:
        log(f"[-] Output validation FAILED: {errs}")
        return False


def main():
    log("=== Boyko Dual-Pass Watch & Push Started ===")
    # Check if already queued or running
    res = subprocess.run(["kaggle", "kernels", "status", KERNEL_SLUG], capture_output=True, text=True)
    status_text = res.stdout + "\n" + res.stderr
    if "QUEUED" in status_text or "RUNNING" in status_text:
        log(f"[*] Kernel is already active: {status_text.strip()}")
    else:
        if not push_kernel():
            sys.exit(1)

    if not monitor_kernel():
        log("[!] Kernel did not complete successfully.")
        sys.exit(1)

    if verify_outputs():
        log("[+] ALL CHECKS PASSED. Ready for Oct 31 competition submission!")
    else:
        log("[-] Output verification failed. Plan recommends fallback to v8 re-roll.")


if __name__ == "__main__":
    main()

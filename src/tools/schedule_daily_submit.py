#!/usr/bin/env python3
"""Autonomous Daily Submission Scheduler for ARC Prize 2026.

Monitors competition submission limits and immediately submits the staged
kernel output as soon as the daily reset occurs at 00:00 UTC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent.parent


def check_remaining_submissions(comp_slug: str = "arc-prize-2026-arc-agi-2") -> int:
    try:
        res = subprocess.run(
            ["kaggle", "competitions", "submission-limits", "-c", comp_slug],
            capture_output=True,
            text=True,
            timeout=30,
        )
        for line in res.stdout.splitlines():
            if "Remaining today:" in line:
                return int(line.split(":")[1].strip())
    except Exception as e:
        print(f"[!] Warning: failed to check submission limits: {e}")
    return 0


def schedule_and_submit(
    kernel_slug: str = "aleixlopez/arc-agi2-hybrid-sota-symbolic-ttt",
    version: int = 7,
    comp_slug: str = "arc-prize-2026-arc-agi-2",
    poll_interval: int = 60,
    max_wait_hours: float = 12.0,
    message: str | None = None,
):
    log_file = ROOT / "schedule_daily_submit.log"
    msg_header = f"[*] Autonomous Scheduler started for [{kernel_slug}] v{version} at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
    print(msg_header)
    with open(log_file, "a") as f:
        f.write(f"\n{msg_header}\n")

    t_start = time.time()
    max_wait_sec = max_wait_hours * 3600

    while time.time() - t_start < max_wait_sec:
        remaining = check_remaining_submissions(comp_slug)
        utc_now = time.strftime("%H:%M:%S UTC", time.gmtime())
        status_line = f"[{utc_now}] Remaining submissions: {remaining}"
        print(status_line)
        with open(log_file, "a") as f:
            f.write(f"{status_line}\n")

        if remaining > 0:
            if message:
                submit_msg = message
            elif version == 13:
                submit_msg = f"ARC-AGI-2 Hybrid SOTA v13 (score_ensemble + 33 Primitives - Record Val 5.0)"
            else:
                submit_msg = f"ARC-AGI-2 Hybrid SOTA v{version} (Kaleidoscope + Kronecker + 1x1 Classifiers + Genus)"
            print(f"[+] Daily quota available! Submitting v{version}: '{submit_msg}'...")
            with open(log_file, "a") as f:
                f.write(f"[+] Submitting v{version} at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")

            sub_res = subprocess.run(
                [
                    "kaggle",
                    "competitions",
                    "submit",
                    "-c",
                    comp_slug,
                    "-k",
                    kernel_slug,
                    "-v",
                    str(version),
                    "-f",
                    "submission.json",
                    "-m",
                    submit_msg,
                ],
                capture_output=True,
                text=True,
            )
            out_str = sub_res.stdout.strip() or sub_res.stderr.strip()
            print(f"[+] Submission result:\n{out_str}")
            with open(log_file, "a") as f:
                f.write(f"[+] Submission result:\n{out_str}\n")

            # Check submissions list
            time.sleep(5)
            list_res = subprocess.run(
                ["kaggle", "competitions", "submissions", "-c", comp_slug],
                capture_output=True,
                text=True,
            )
            print(f"[*] Updated Submissions Queue:\n{list_res.stdout.strip()}")
            with open(log_file, "a") as f:
                f.write(f"[*] Updated Submissions Queue:\n{list_res.stdout.strip()}\n")
            return True

        time.sleep(poll_interval)

    print("[!] Maximum wait time exceeded in scheduler.")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="aleixlopez/arc-agi2-hybrid-sota-symbolic-ttt")
    parser.add_argument("--version", type=int, default=13)
    parser.add_argument("--message", default=None, help="Custom submission message")
    parser.add_argument("--comp", default="arc-prize-2026-arc-agi-2")
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--max-wait", type=float, default=24.0)
    args = parser.parse_args()
    schedule_and_submit(
        kernel_slug=args.kernel,
        version=args.version,
        comp_slug=args.comp,
        poll_interval=args.poll_interval,
        max_wait_hours=args.max_wait,
        message=args.message,
    )

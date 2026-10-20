from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.validator import validate_submission_dict


def get_kernel_status(kernel_slug: str) -> str:
    res = subprocess.run(
        ["kaggle", "kernels", "status", kernel_slug],
        capture_output=True,
        text=True,
    )
    for line in res.stdout.splitlines():
        if "has status" in line:
            return line.split('"')[1]
    return "UNKNOWN"


def check_daily_submission_limit(comp_slug: str) -> int:
    res = subprocess.run(
        ["kaggle", "competitions", "submission-limits", "-c", comp_slug],
        capture_output=True,
        text=True,
    )
    for line in res.stdout.splitlines():
        if "Remaining today:" in line:
            try:
                return int(line.split(":")[1].strip())
            except Exception:
                pass
    return 0


def watch_and_submit(
    kernel_slug: str = "aleixlopez/arc-agi2-hybrid-sota-symbolic-ttt",
    version: int = 5,
    comp_slug: str = "arc-prize-2026-arc-agi-2",
    poll_interval_sec: int = 45,
    max_wait_hours: float = 3.0,
):
    log_file = ROOT / "auto_submit.log"
    print(f"[*] Starting Auto-Submit Watcher for kernel [{kernel_slug}] (v{version})")
    print(f"[*] Target competition: {comp_slug}")
    with open(log_file, "a") as f_log:
        f_log.write(f"\n--- Started watcher for {kernel_slug} v{version} at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    t_start = time.time()
    max_wait_sec = max_wait_hours * 3600

    while time.time() - t_start < max_wait_sec:
        try:
            status = get_kernel_status(kernel_slug)
            elapsed_min = (time.time() - t_start) / 60
            log_line = f"[{time.strftime('%H:%M:%S')}] Elapsed: {elapsed_min:.1f}m | Kernel Status: {status}\n"
            print(log_line, end="")
            with open(log_file, "a") as f_log:
                f_log.write(log_line)

            if status == "KernelWorkerStatus.COMPLETE":
                print(f"[+] Kernel [{kernel_slug}] has COMPLETED successfully!")
                with open(log_file, "a") as f_log:
                    f_log.write(f"[+] Kernel completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

                # 1. Download output to verify
                dl_dir = Path("/tmp/auto_sub_output")
                dl_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    ["kaggle", "kernels", "output", kernel_slug, "-p", str(dl_dir)],
                    capture_output=True,
                )

                sub_path = dl_dir / "submission.json"
                if not sub_path.exists():
                    print(f"[!] ERROR: submission.json not found in output directory {dl_dir}")
                    return False

                # 2. Validate schema
                sub_data = json.loads(sub_path.read_text())
                eval_chal = json.loads(config.EVAL_CHALLENGES_PATH.read_text())
                ok, errors = validate_submission_dict(sub_data, eval_chal)
                if not ok:
                    print(f"[!] Validation failed with errors: {errors[:5]}")
                    return False
                print(f"[+] Output verified: 100% compliant with schema across {len(sub_data)} tasks.")

                # Cache local verified submission
                cache_dest = ROOT / "submissions" / f"submission_hybrid_sota_v{version}.json"
                cache_dest.write_text(json.dumps(sub_data, indent=2))
                print(f"[+] Cached verified submission to {cache_dest}")

                # 3. Check remaining daily limit
                remaining = check_daily_submission_limit(comp_slug)
                print(f"[*] Remaining daily submissions: {remaining}")
                if remaining <= 0:
                    print(f"[!] Daily submission quota reached. Handing off to schedule_daily_submit for v{version} at 00:00 UTC.")
                    from src.tools.schedule_daily_submit import schedule_and_submit
                    return schedule_and_submit(
                        kernel_slug=kernel_slug,
                        version=version,
                        comp_slug=comp_slug,
                    )

                # 4. Fire competition submission
                if version == 25:
                    msg = f"ARC-AGI-2 Hybrid SOTA v{version} (Dual-Pass Pooling Seed 1+137 + 42 Primitives - Record Val 9.30%)"
                elif version == 24:
                    msg = f"ARC-AGI-2 Hybrid SOTA v{version} (Optical Ray Mirrors + Cheap-First - Record Val 9.30%)"
                elif version == 23:
                    msg = f"ARC-AGI-2 Hybrid SOTA v{version} (Ascending Pillar Sorting + Cheap-First - Record Val 8.72%)"
                elif version == 22:
                    msg = f"ARC-AGI-2 Hybrid SOTA v{version} (Cheap-First Work Scheduling + Cross-Family Pass@2 - 40 Primitives)"
                elif version == 21:
                    msg = f"ARC-AGI-2 Hybrid SOTA v{version} (Cross-Family Pass@2 Selection + 40 Primitives - Record Val 12.0)"
                elif version == 20:
                    msg = f"ARC-AGI-2 Hybrid SOTA v{version} (Multi-Room Partition Overlay + 40 Primitives - Record Val 8.14)"
                elif version == 19 or version == 18:
                    msg = f"ARC-AGI-2 Hybrid SOTA v{version} (Sokoban Channel Sliding + Neural Invariant Pruning - Record Val 7.56)"
                elif version == 17:
                    msg = f"ARC-AGI-2 Hybrid SOTA v{version} (Deficit-First GPU Scheduling + CRC32 Determinism - Record Val 10.0)"
                elif version == 16:
                    msg = f"ARC-AGI-2 Hybrid SOTA v{version} (Liquid Gravity Flow + Parallel Line Standardization - Record Val 6.98)"
                elif version == 15:
                    msg = f"ARC-AGI-2 Hybrid SOTA v{version} (Parallel Line Standardization + 37 Primitives - Record Val 6.40)"
                elif version == 14:
                    msg = f"ARC-AGI-2 Hybrid SOTA v{version} (Symmetry Restoration + Ranked Pillars + Legend Cavity - Record Val 5.81)"
                elif version == 13:
                    msg = f"ARC-AGI-2 Hybrid SOTA v{version} (score_ensemble + 33 Primitives - Record Val 5.0)"
                else:
                    msg = f"ARC-AGI-2 Hybrid SOTA v{version} (Optimal Union + 38 Symbolic Primitives + Deep Eval)"
                print(f"[*] Firing competition submission: '{msg}'...")
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
                        msg,
                    ],
                    capture_output=True,
                    text=True,
                )
                print(f"[+] Submission output: {sub_res.stdout.strip()}")
                with open(log_file, "a") as f_log:
                    f_log.write(f"[+] Submission output: {sub_res.stdout.strip()}\n")

                # 5. Check submissions list
                time.sleep(5)
                list_res = subprocess.run(
                    ["kaggle", "competitions", "submissions", "-c", comp_slug],
                    capture_output=True,
                    text=True,
                )
                print(f"[*] Live Submissions Queue:\n{list_res.stdout.strip()}")
                return True

            elif status == "KernelWorkerStatus.ERROR":
                print(f"[!] Kernel [{kernel_slug}] encountered an ERROR. Aborting auto-submit.")
                return False

        except Exception as e:
            print(f"[!] Exception during watch iteration: {e}")

        time.sleep(poll_interval_sec)

    print("[!] Maximum wait time exceeded. Exiting watcher.")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="aleixlopez/arc-agi2-hybrid-sota-symbolic-ttt")
    parser.add_argument("--version", type=int, default=14)
    parser.add_argument("--poll-interval", type=int, default=45)
    parser.add_argument("--max-wait", type=float, default=8.0)
    args = parser.parse_args()
    watch_and_submit(
        kernel_slug=args.kernel,
        version=args.version,
        poll_interval_sec=args.poll_interval,
        max_wait_hours=args.max_wait,
    )

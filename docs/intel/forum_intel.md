# Forum Discussions & Community Intel: arc-prize-2026-arc-agi-2

| ID | Title | Votes | Comments | Date |
| :--- | :--- | :--- | :--- | :--- |
| 689054 | Upgraded Accelerators | 21 | 10 | 2026-04-07 16:31:11.629000 |
| 689254 | ARC Prize 2026 - ARC-AGI-2 - Let's go! | 8 | 7 | 2026-04-08 02:26:47.245000 |
| 684441 | How to get started + Competition's Official Discord | 4 | 1 | 2026-03-24 21:16:14.165000 |
| 709522 | Historical leaderboard with runtime information | 30 | 6 | 2026-06-19 03:27:19.292000 |
| 685142 | Last Years 1st Place Solution | 25 | 21 | 2026-03-27 00:29:52.980000 |
| 694698 | 95% on ARC AGI1 evaluation | 20 | 2 | 2026-04-26 12:08:12.016000 |
| 738313 | Weird timeout | 14 | 8 | 2026-08-31 08:25:14.409000 |
| 724647 | Good to see movement on LB | 13 | 17 | 2026-07-12 13:03:23.331000 |
| 691081 | When will a new model be top? | 11 | 18 | 2026-04-13 19:46:35.083000 |
| 729743 | Open-sourcing my work and analysis until now | 11 | 4 | 2026-07-26 13:51:26.775000 |
| 697859 | Bitterbot's TOPAS Architecture Submission: 11.67% LB vs ~36% Local (Managing TTT Timeouts) | 10 | 23 | 2026-05-07 16:48:56.703000 |
| 743927 | These 2 days the notebooks have been easily queued for several hours and couldn't be executed immediately | 9 | 12 | 2026-09-27 19:16:14.448000 |
| 687095 | when will we get gpu L4x4 | 8 | 3 | 2026-04-02 07:24:05.513000 |
| 698462 | task  17829a00 | 8 | 7 | 2026-05-10 03:01:37.995000 |
| 684662 | Discrepancy between stated prizes on Kaggle page & ARC-AGI-2 2026 page? | 5 | 7 | 2026-03-25 18:43:54.428000 |
| 687701 | What is ARC actually measuring? | 5 | 5 | 2026-04-03 13:33:43.261000 |
| 694653 | [Open Source] 400/400 ARC-1 Training Dataset Solver Released! | 5 | 1 | 2026-04-26 06:41:42.147000 |
| 732723 | (PUBLIC) Nex-N2-mini vs Aquila-mini: two 35B post-trains on ARC-AGI-2, head to head | 5 | 2 | 2026-08-04 11:43:51.258000 |
| 731040 | does task 963c33f8 contains an error ? | 5 | 5 | 2026-07-30 17:14:17.309000 |
| 697058 | Funny observation | 4 | 6 | 2026-05-05 05:24:59.299000 |
| 753674 | Request for Deadline Extension or Rerun of Failed/Final Submissions | 25 | 9 | 2026-10-28 23:18:01.122000 |
| 753659 | 3rd day without working sub | 24 | 13 | 2026-10-28 22:04:24.443000 |
| 750577 | Watching LLM Agents Solve ARC-AGI-2 | 24 | 10 | 2026-10-19 00:10:47.398000 |
| 747515 | Update on "Kaggle Error" Submissions & Queue Delays | 21 | 37 | 2026-10-08 22:04:06.132000 |
| 750546 | Qwen3.8 or something else? | 17 | 38 | 2026-10-18 19:43:49.856000 |
| 752531 | My submission has been running for the past 19 hours | 16 | 14 | 2026-10-25 08:17:35.666000 |
| 753723 | Scoring 24h+ — proposal if delays persist | 15 | 5 | 2026-10-29 04:34:11.346000 |
| 753315 | Unstable GPU server | 11 | 18 | 2026-10-27 20:39:58.678000 |
| 752709 | Are solutions using Qwen3.8-Flash-Next eligible for prizes? | 11 | 2 | 2026-10-26 01:18:51.067000 |
| 748082 | Qwen3.8-27B / Flash-Next: ARC-AGI-2 results within Kaggle’s compute limits? | 7 | 33 | 2026-10-10 14:47:28.908000 |
| 753820 | Some data points on the current unknown Kaggle platform issue | 7 | 2 | 2026-10-29 12:30:32.098000 |
| 753865 | Significant submission score drop not reproduced locally | 5 | 2 | 2026-10-29 16:03:18.456000 |
| 748741 | Are Codex- or Claude-generated synthetic ARC puzzles allowed for model training? | 4 | 4 | 2026-10-12 23:51:48.700000 |
| 751292 | Question on Grand Prize and Private Leaderboard | 1 | 22 | 2026-10-21 09:25:56.618000 |

## High-Value Technical Discussions

*(For the complete comprehensive late-stage forum audit, see `docs/intel/unconsidered_and_recent_discussions_audit.md`)*

### Topic 750546: Qwen3.8 or something else? (The 55%+ Public LB Surge)
- **Context:** Community discussion deconstructing the jump from ~35% to 55-60%+ on the public leaderboard.
- **Key Discovery:** Leading teams adopted **Qwen3.8-Flash-Next** and **Qwen3.8-27B** wrapped in an iterative Python REPL execution harness (the "Duck harness", pioneered in Daniel Franzen's ARC-AGI-3 milestone solution).
- **Ensemble Gain:** Participants reported lifting scores from 35% to 45% simply by ensembling their existing NVARC pipeline with a Qwen3.8-27B run.
- **Compute Reality:** A 27B model generating ~70k tokens median per task requires 365–440 tok/s on 4x L4 to cover all 240 test tasks in 12 hours. Teams without aggressive optimization or early exit hit global timeouts.

### Topic 753315, 753820, 753865: Kaggle Cluster Instability & 10x Inter-GPU Bottleneck
- **Hardware Contention:** Saturated 4x L4 cluster due to concurrent ARC-AGI-2 and Google Gemma 4 Agent competitions.
- **Severe Node Variance:** Certain 4x L4 nodes exhibit up to **10x slower inter-GPU communication bandwidth**, crippling multi-GPU communication and causing timeouts.
- **Score Volatility:** Teams observed unprovoked drops from 56.94% to 26.67% or 0.00 without code changes due to timeouts on slow nodes.
- **Protection:** Independent single-GPU workers (no inter-GPU synchronization) and strict per-task wall-clock timeouts (150s cap) with fallback.

### Topic 751292: Critical Deadline Rule Clarification
- **Deadline:** November 2, 2026, 23:59 UTC.
- **Rule Clarification:** Kaggle staff corrected an earlier comment: **pending submissions that are still scoring when the deadline passes cannot be safely selected for final evaluation.**
- **Action:** Final submissions must be launched at least 36–48 hours before the deadline to accommodate 24h+ queue wait times.

### Topic 694698: 95% on ARC AGI1 evaluation

<p>After last year competition, we ran our code on ARC AGI1 public evaluation data, and we scored above 95%.</p>
<p>The code to reproduce it is in our last year solution github repo: <a href="https://github.com/1ytic/NVARC/tree/main/ARC-AGI1" target="_blank">https://github.com/1ytic/NVARC/tree/main/ARC-AGI1</a></p>
<p>The reason for the high score is twofold:</p>
<ol>
<li>ARC AGI1 puzzles are much easier than ARC AGI2.</li>
<li>Most of ARC AGI1 evaluation is included in ARC AGI2 training data.</li>
</ol>
<p>The second reason is the most important one. It shows that ARC AGI1 scores need to be taken with a grain of salt, depending on the use of ARC AGI2 data or not. All frontier models use ARC AGI2 training data, hence their scores on ARC AGI1 are meaningless.</p>
<p>I posted this because I see some of you posting ARC AGI1 evaluation scores. This maybe a good first step, but you should move to ARC AGI2 as soon as possible; it is much harder.</p>

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

## High-Value Technical Discussions

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

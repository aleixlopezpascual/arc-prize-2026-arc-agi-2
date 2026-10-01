# Evaluation Metric & Submission: arc-prize-2026-arc-agi-2

This competition evaluates submissions on the percentage of correct predictions. For each task, you should make 2 attempts to predict the exact outputs for every test input grid contained in the task. (Tasks can have more than one test input that needs a predicted output.) Each task test output has one ground truth. For a given task output, any of the 2 predicted outputs matches the ground truth exactly, you score `1` for that task test output, otherwise `0`. The final score is the sum averaged of the highest score per task output divided by the total number of task test outputs.
## Submission File
The submission file for this competition must be a json named `submission.json`. 
For each task output in the evaluation set, you should make exactly 2 predictions (`attempt_1`, `attempt_2`). The structure of predictions is shown below. Many tasks have multiple outputs (a multiple dictionaries enclosed in a list), although some tasks have a single output that must be predicted. When a task has multiple test outputs that need to be predicted (e.g., task `12997ef3` below), they must be in the same order as the corresponding test inputs.
**IMPORTANT:** All the task_ids in the input challenges json file must also be present in the `submission.json` file. Both "attempt_1" and "attempt_2" must be present, even if your submission doesn't have 2 predictions.
    {"00576224": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}],
     "009d5c81": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}],
     "12997ef3": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]},
 	             {"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}],
     ...
    }
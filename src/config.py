from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
SRC_DIR = ROOT / "src"
DOCS_DIR = ROOT / "docs"
OOF_DIR = ROOT / "oof"
SUBMISSION_DIR = ROOT / "submissions"
MODELS_DIR = ROOT / "models"

# Dataset file paths
TRAIN_CHALLENGES_PATH = INPUT_DIR / "arc-agi_training_challenges.json"
TRAIN_SOLUTIONS_PATH = INPUT_DIR / "arc-agi_training_solutions.json"
EVAL_CHALLENGES_PATH = INPUT_DIR / "arc-agi_evaluation_challenges.json"
EVAL_SOLUTIONS_PATH = INPUT_DIR / "arc-agi_evaluation_solutions.json"
TEST_CHALLENGES_PATH = INPUT_DIR / "arc-agi_test_challenges.json"
SAMPLE_SUBMISSION_PATH = INPUT_DIR / "sample_submission.json"

# Competition parameters
MAX_ATTEMPTS = 2
MIN_GRID_DIM = 1
MAX_GRID_DIM = 30
NUM_COLORS = 10
SEED = 42
DEFAULT_TASK_TIMEOUT_SEC = 120

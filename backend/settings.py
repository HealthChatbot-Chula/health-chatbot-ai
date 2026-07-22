from pathlib import Path


HEALTH_MODEL = "health-agent"
EVAL_SIMULATOR_MODEL = "health-eval-simulator"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USER_SIMULATION_CASES_PATH = PROJECT_ROOT / "eval" / "user_simulation_cases.json"
JUDGE_CRITERIA_PATH = PROJECT_ROOT / "eval" / "judge_criteria.json"

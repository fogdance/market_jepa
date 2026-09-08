"""One canonical, human-reviewed RB campaign per repository deployment."""
import json
from pathlib import Path

from .protocol import file_hash

CAMPAIGN_ROOT = Path(__file__).resolve().parents[2] / "artifacts/evaluation/v1_1_formal_campaign"


def campaign_plan(path, freeze_path):
    root = CAMPAIGN_ROOT.resolve()
    if Path(path).resolve() != root / "campaign_plan.json" or Path(freeze_path).resolve() != root / "rb_freeze.json":
        raise ValueError("RB campaign must use canonical plan/freeze paths")
    if (root / "rb_test_consumption.json").exists():
        raise FileExistsError("canonical RB-Test campaign already consumed")
    plan = json.loads(Path(path).read_text())
    expected = [(m["model_size"], m["variant"]) for m in plan["expected_models"]]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("campaign expected cohort empty/duplicated")
    return expected, file_hash(path)


def exact_cohort(actual, expected):
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("actual RB cohort does not equal campaign expected cohort")

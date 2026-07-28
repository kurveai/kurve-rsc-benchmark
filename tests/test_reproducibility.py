from scripts.run_all import CLASSIFICATION_TASKS, REGRESSION_TASKS, TASK_GROUPS

from kurve_rsc.feature_pipeline import _incremental_model_params
from kurve_rsc.relbench_regression_metrics import normalized_mae


def test_v1_task_registry_is_complete_and_unique():
    tasks = TASK_GROUPS["v1"]
    assert len(tasks) == 21
    assert len(set(tasks)) == len(tasks)
    assert len(CLASSIFICATION_TASKS) == 12
    assert len(REGRESSION_TASKS) == 9


def test_incremental_model_parameters_are_deterministic():
    params = _incremental_model_params(
        {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 10.0},
        "Logloss",
        10,
    )
    assert params["random_seed"] == 42
    assert params["allow_writing_files"] is False


def test_nmae_uses_only_the_training_target_standard_deviation():
    assert normalized_mae([1.0, 3.0], [2.0, 2.0], [1.0, 3.0]) == 1.0

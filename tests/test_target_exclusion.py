import pandas as pd

from kurve_rsc.feature_pipeline import fit_incremental_classifier


def test_incremental_training_receives_only_explicit_model_features():
    batch = pd.DataFrame(
        {
            "signal": [0.0, 1.0, 0.2, 0.8],
            "target": [0, 1, 0, 1],
            "label_leak": [0, 1, 0, 1],
        }
    )
    model, _ = fit_incremental_classifier(
        lambda: iter([batch]),
        ["signal"],
        "target",
        batch[["signal"]],
        batch["target"],
        batch_count=1,
        config={
            "iterations": 4,
            "depth": 2,
            "learning_rate": 0.1,
            "l2_leaf_reg": 3.0,
        },
    )

    assert model.feature_names_ == ["signal"]

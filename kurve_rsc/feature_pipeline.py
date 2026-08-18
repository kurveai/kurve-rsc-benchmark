"""Model-fitting helpers for the RelBench examples."""

from __future__ import annotations

import os
from math import ceil
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import mean_absolute_error, roc_auc_score


MODEL_BACKEND_ENV = "KURVE_RSC_MODEL_BACKEND"
TRAIN_ALL_AT_ONCE_ENV = "KURVE_RSC_TRAIN_ALL_AT_ONCE"
TABPFN_CONFIG = {"name": "tabpfn"}

CLASSIFIER_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "depth6_regularized",
        "iterations": 2000,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 10.0,
    },
    {
        "name": "depth7_regularized",
        "iterations": 1500,
        "depth": 7,
        "learning_rate": 0.03,
        "l2_leaf_reg": 10.0,
    },
    {
        "name": "depth5_regularized",
        "iterations": 2500,
        "depth": 5,
        "learning_rate": 0.02,
        "l2_leaf_reg": 12.0,
    },
    {
        "name": "depth6_fast",
        "iterations": 1000,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 12.0,
    },
)

REGRESSOR_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "depth6_regularized",
        "iterations": 2000,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 10.0,
    },
    {
        "name": "depth7_regularized",
        "iterations": 1500,
        "depth": 7,
        "learning_rate": 0.03,
        "l2_leaf_reg": 10.0,
    },
    {
        "name": "depth5_regularized",
        "iterations": 2500,
        "depth": 5,
        "learning_rate": 0.02,
        "l2_leaf_reg": 12.0,
    },
    {
        "name": "depth6_fast",
        "iterations": 1000,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 12.0,
    },
)

ALL_FEATURE_FAMILIES = (
    "base",
    "semantic",
    "conditional",
    "temporal",
    "sequence",
    "episode",
    "context",
)
TEMPORAL_FEATURE_FAMILIES = (
    "base",
    "semantic",
    "conditional",
    "temporal",
    "sequence",
    "episode",
    "context",
)


def selected_model_backend(
    default: str = "catboost",
    override: str | None = None,
) -> str:
    """Return the requested model backend and reject unsupported values."""

    backend = (
        override
        if override is not None
        else os.environ.get(MODEL_BACKEND_ENV, default)
    ).strip().lower()
    if backend not in {"catboost", "tabpfn"}:
        raise ValueError(
            f"{MODEL_BACKEND_ENV} must be either 'catboost' or 'tabpfn'"
        )
    return backend


def train_all_at_once_enabled(override: bool | None = None) -> bool:
    """Return whether disk-backed training frames should be fit jointly."""

    if override is not None:
        return bool(override)
    raw_value = os.environ.get(TRAIN_ALL_AT_ONCE_ENV, "0").strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{TRAIN_ALL_AT_ONCE_ENV} must be a boolean value")


def enable_all_feature_families(nodes: Iterable[Any]) -> None:
    """Enable every SQL feature family on a graph's nodes."""

    for node in nodes:
        node.feature_families = ALL_FEATURE_FAMILIES


def set_feature_families(nodes: Iterable[Any], families: Sequence[str]) -> None:
    """Select SQL auto-feature families without adding custom feature logic."""

    selected = tuple(dict.fromkeys(families))
    unknown = set(selected) - set(ALL_FEATURE_FAMILIES)
    if unknown:
        raise ValueError(f"Unknown SQL auto-feature families: {sorted(unknown)}")
    for node in nodes:
        node.feature_families = selected


def fit_tuned_classifier(
    train_inputs: pd.DataFrame,
    train_target: pd.Series,
    val_inputs: pd.DataFrame,
    val_target: pd.Series,
    *,
    cat_features: Sequence[int] | None = None,
    auto_class_weights: str | None = None,
    configs: Sequence[dict[str, Any]] = CLASSIFIER_CONFIGS,
    model_backend: str | None = None,
) -> tuple[Any, dict[str, Any], float]:
    """Select a classifier by validation AUROC without touching the test split."""

    if selected_model_backend(override=model_backend) == "tabpfn":
        batch = train_inputs.reindex(columns=list(train_inputs.columns)).copy()
        target_column = "__tabpfn_target__"
        batch[target_column] = train_target.to_numpy()
        model, auc, _ = fit_tabpfn_classifier(
            lambda: iter([batch]),
            list(train_inputs.columns),
            target_column,
            val_inputs,
            val_target,
            cat_features=cat_features,
        )
        return model, dict(TABPFN_CONFIG), auc

    best_model: CatBoostClassifier | None = None
    best_config: dict[str, Any] | None = None
    best_auc = float("-inf")
    for config in configs:
        params = {
            "iterations": int(config["iterations"]),
            "depth": int(config["depth"]),
            "learning_rate": float(config["learning_rate"]),
            "l2_leaf_reg": float(config["l2_leaf_reg"]),
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": 42,
            "random_strength": 1.0,
            "bagging_temperature": 1.0,
            "verbose": False,
            "allow_writing_files": False,
        }
        if auto_class_weights is not None:
            params["auto_class_weights"] = auto_class_weights
        model = CatBoostClassifier(**params)
        model.fit(
            train_inputs,
            train_target,
            cat_features=cat_features,
            eval_set=(val_inputs, val_target),
            use_best_model=True,
            early_stopping_rounds=200,
        )
        predictions = model.predict_proba(val_inputs)[:, 1]
        auc = float(roc_auc_score(val_target.to_numpy(dtype="float64"), predictions))
        if auc > best_auc:
            best_model = model
            best_config = config
            best_auc = auc

    if best_model is None or best_config is None:
        raise RuntimeError("CatBoost classifier tuning produced no model")
    return best_model, best_config, best_auc


def fit_tuned_regressor(
    train_inputs: pd.DataFrame,
    train_target: pd.Series,
    val_inputs: pd.DataFrame,
    val_target: pd.Series,
    *,
    cat_features: Sequence[int] | None = None,
    configs: Sequence[dict[str, Any]] = REGRESSOR_CONFIGS,
    model_backend: str | None = None,
) -> tuple[Any, dict[str, Any], float]:
    """Select a regressor by validation MAE without touching the test split."""

    if selected_model_backend(override=model_backend) == "tabpfn":
        batch = train_inputs.reindex(columns=list(train_inputs.columns)).copy()
        target_column = "__tabpfn_target__"
        batch[target_column] = train_target.to_numpy()
        model, mae, _ = fit_tabpfn_regressor(
            lambda: iter([batch]),
            list(train_inputs.columns),
            target_column,
            val_inputs,
            val_target,
        )
        return model, dict(TABPFN_CONFIG), mae

    best_model: CatBoostRegressor | None = None
    best_config: dict[str, Any] | None = None
    best_mae = float("inf")
    for config in configs:
        model = CatBoostRegressor(
            iterations=int(config["iterations"]),
            depth=int(config["depth"]),
            learning_rate=float(config["learning_rate"]),
            l2_leaf_reg=float(config["l2_leaf_reg"]),
            loss_function="MAE",
            eval_metric="MAE",
            random_seed=42,
            random_strength=1.0,
            bagging_temperature=1.0,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            train_inputs,
            train_target,
            cat_features=cat_features,
            eval_set=(val_inputs, val_target),
            use_best_model=True,
            early_stopping_rounds=200,
        )
        predictions = np.asarray(model.predict(val_inputs), dtype="float64")
        mae = float(mean_absolute_error(val_target.to_numpy(dtype="float64"), predictions))
        if mae < best_mae:
            best_model = model
            best_config = config
            best_mae = mae

    if best_model is None or best_config is None:
        raise RuntimeError("CatBoost regressor tuning produced no model")
    return best_model, best_config, best_mae


def fit_tuned_cross_entropy_model(
    train_inputs: pd.DataFrame,
    train_target: pd.Series,
    val_inputs: pd.DataFrame,
    val_target: pd.Series,
    *,
    cat_features: Sequence[int] | None = None,
    configs: Sequence[dict[str, Any]] = REGRESSOR_CONFIGS,
) -> tuple[CatBoostClassifier, dict[str, Any], float]:
    """Fit a probability model for a fractional target in ``[0, 1]``.

    CatBoost's ``CrossEntropy`` objective accepts soft labels, unlike
    ``Logloss``. Configuration selection remains aligned with RelBench by
    comparing the resulting probabilities with validation MAE.
    """

    train_values = train_target.to_numpy(dtype="float64")
    val_values = val_target.to_numpy(dtype="float64")
    if (
        not np.isfinite(train_values).all()
        or not np.isfinite(val_values).all()
        or np.any((train_values < 0.0) | (train_values > 1.0))
        or np.any((val_values < 0.0) | (val_values > 1.0))
    ):
        raise ValueError("Cross-entropy targets must be finite and lie in [0, 1]")

    best_model: CatBoostClassifier | None = None
    best_config: dict[str, Any] | None = None
    best_mae = float("inf")
    for config in configs:
        model = CatBoostClassifier(
            iterations=int(config["iterations"]),
            depth=int(config["depth"]),
            learning_rate=float(config["learning_rate"]),
            l2_leaf_reg=float(config["l2_leaf_reg"]),
            loss_function="CrossEntropy",
            eval_metric="CrossEntropy",
            random_seed=42,
            random_strength=1.0,
            bagging_temperature=1.0,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            train_inputs,
            train_target,
            cat_features=cat_features,
            eval_set=(val_inputs, val_target),
            use_best_model=True,
            early_stopping_rounds=200,
        )
        predictions = np.asarray(
            model.predict_proba(val_inputs)[:, 1],
            dtype="float64",
        )
        mae = float(mean_absolute_error(val_values, predictions))
        if mae < best_mae:
            best_model = model
            best_config = config
            best_mae = mae

    if best_model is None or best_config is None:
        raise RuntimeError("CatBoost cross-entropy tuning produced no model")
    return best_model, best_config, best_mae


def prepare_tabpfn_inputs(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Build a compact numeric frame while preserving missing values for TabPFN."""

    return (
        frame.reindex(columns=list(feature_columns), fill_value=np.nan)
        .replace([np.inf, -np.inf], np.nan)
        .astype("float32")
    )


class _TabPFNPreprocessor:
    """Encode mixed task frames consistently for TabPFN."""

    def __init__(
        self,
        feature_columns: Sequence[str],
        categorical_maps: dict[str, dict[str, int]],
    ) -> None:
        self.feature_columns = tuple(feature_columns)
        self.categorical_maps = categorical_maps
        self.categorical_features_indices = [
            index
            for index, column in enumerate(self.feature_columns)
            if column in categorical_maps
        ]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        feature_columns: Sequence[str],
        categorical_features_indices: Sequence[int] | None = None,
    ) -> "_TabPFNPreprocessor":
        forced_categorical = set(categorical_features_indices or ())
        categorical_maps: dict[str, dict[str, int]] = {}
        for index, column in enumerate(feature_columns):
            series = frame[column]
            if index not in forced_categorical and is_numeric_dtype(series.dtype):
                continue
            values = series.astype("string").dropna().drop_duplicates()
            categorical_maps[column] = {
                str(value): category for category, value in enumerate(values)
            }
        return cls(feature_columns, categorical_maps)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = pd.DataFrame(index=frame.index)
        for column in self.feature_columns:
            if column in frame:
                series = frame[column]
            else:
                series = pd.Series(np.nan, index=frame.index)
            category_map = self.categorical_maps.get(column)
            if category_map is None:
                transformed[column] = pd.to_numeric(series, errors="coerce")
            else:
                transformed[column] = series.astype("string").map(category_map)
        return (
            transformed.replace([np.inf, -np.inf], np.nan)
            .astype("float32")
            .reset_index(drop=True)
        )


class _TabPFNModelAdapter:
    """Apply the fitted task preprocessor before TabPFN inference."""

    def __init__(self, estimator: Any, preprocessor: _TabPFNPreprocessor) -> None:
        self.estimator = estimator
        self.preprocessor = preprocessor

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            self.estimator.predict(self.preprocessor.transform(inputs))
        )

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            self.estimator.predict_proba(self.preprocessor.transform(inputs))
        )

    def get_best_iteration(self) -> int:
        """Match the small CatBoost interface used by task diagnostics."""

        return -1

    def __getattr__(self, name: str) -> Any:
        return getattr(self.estimator, name)


def _materialize_tabpfn_training_data(
    train_batch_factory: Callable[[], Iterator[pd.DataFrame]],
    feature_columns: Sequence[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    train_inputs: list[pd.DataFrame] = []
    train_targets: list[pd.Series] = []
    for batch in train_batch_factory():
        if batch.empty:
            continue
        train_inputs.append(batch.reindex(columns=list(feature_columns)))
        train_targets.append(batch[target_column].reset_index(drop=True))
    if not train_inputs:
        raise RuntimeError("TabPFN received no training rows")
    return (
        pd.concat(train_inputs, ignore_index=True),
        pd.concat(train_targets, ignore_index=True),
    )


def fit_tabpfn_classifier(
    train_batch_factory: Callable[[], Iterator[pd.DataFrame]],
    feature_columns: Sequence[str],
    target_column: str,
    val_inputs: pd.DataFrame,
    val_target: pd.Series,
    *,
    cat_features: Sequence[int] | None = None,
    classifier_factory: Callable[..., Any] | None = None,
) -> tuple[Any, float, np.ndarray]:
    """Fit a TabPFN classifier on materialized training batches."""

    print("model_backend: tabpfn", flush=True)
    raw_train_inputs, train_target = _materialize_tabpfn_training_data(
        train_batch_factory,
        feature_columns,
        target_column,
    )
    valid_target = train_target.notna()
    raw_train_inputs = raw_train_inputs.loc[valid_target].reset_index(drop=True)
    train_target = train_target.loc[valid_target].reset_index(drop=True)
    if train_target.nunique(dropna=True) < 2:
        raise RuntimeError("TabPFN classifier requires at least two target classes")

    preprocessor = _TabPFNPreprocessor.fit(
        raw_train_inputs,
        feature_columns,
        categorical_features_indices=cat_features,
    )
    if classifier_factory is None:
        from tabpfn import TabPFNClassifier

        classifier_factory = TabPFNClassifier

    model_kwargs: dict[str, Any] = {
        "random_state": 42,
        "ignore_pretraining_limits": True,
    }
    if preprocessor.categorical_features_indices:
        model_kwargs["categorical_features_indices"] = (
            preprocessor.categorical_features_indices
        )
    estimator = classifier_factory(**model_kwargs)
    estimator.fit(preprocessor.transform(raw_train_inputs), train_target)
    model = _TabPFNModelAdapter(estimator, preprocessor)
    predictions = np.asarray(model.predict_proba(val_inputs)[:, 1], dtype="float64")
    auc = float(
        roc_auc_score(val_target.to_numpy(dtype="float64"), predictions)
    )
    return model, auc, predictions


def fit_tabpfn_regressor(
    train_batch_factory: Callable[[], Iterator[pd.DataFrame]],
    feature_columns: Sequence[str],
    target_column: str,
    val_inputs: pd.DataFrame,
    val_target: pd.Series,
    *,
    regressor_factory: Callable[..., Any] | None = None,
) -> tuple[Any, float, np.ndarray]:
    """Fit the local TabPFN regressor on materialized training batches."""

    print("model_backend: tabpfn", flush=True)
    raw_train_inputs, train_target = _materialize_tabpfn_training_data(
        train_batch_factory,
        feature_columns,
        target_column,
    )
    train_target = (
        pd.to_numeric(train_target, errors="coerce").fillna(0).astype("float64")
    )
    preprocessor = _TabPFNPreprocessor.fit(raw_train_inputs, feature_columns)

    if regressor_factory is None:
        from tabpfn import TabPFNRegressor

        regressor_factory = TabPFNRegressor

    model_kwargs: dict[str, Any] = {
        "random_state": 42,
        "ignore_pretraining_limits": True,
    }
    if preprocessor.categorical_features_indices:
        model_kwargs["categorical_features_indices"] = (
            preprocessor.categorical_features_indices
        )
    estimator = regressor_factory(
        **model_kwargs,
    )
    estimator.fit(preprocessor.transform(raw_train_inputs), train_target)
    model = _TabPFNModelAdapter(estimator, preprocessor)
    predictions = np.asarray(
        model.predict(val_inputs),
        dtype="float64",
    )
    mae = float(
        mean_absolute_error(val_target.to_numpy(dtype="float64"), predictions)
    )
    return model, mae, predictions


def _incremental_model_params(config: dict[str, Any], loss_function: str, iterations: int) -> dict[str, Any]:
    return {
        "iterations": iterations,
        "depth": int(config["depth"]),
        "learning_rate": float(config["learning_rate"]),
        "l2_leaf_reg": float(config["l2_leaf_reg"]),
        "loss_function": loss_function,
        "random_seed": 42,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
        "verbose": False,
        "allow_writing_files": False,
    }


def fit_incremental_classifier(
    train_batch_factory: Callable[[], Iterator[pd.DataFrame]],
    feature_columns: Sequence[str],
    target_column: str,
    val_inputs: pd.DataFrame,
    val_target: pd.Series,
    *,
    batch_count: int,
    config: dict[str, Any],
    cat_features: Sequence[int] | None = None,
    auto_class_weights: str | None = None,
    model_backend: str | None = None,
    train_all_at_once: bool | None = None,
    _report_training_mode: bool = True,
) -> tuple[Any, float]:
    """Fit one classifier jointly or by continuing across disk-backed batches."""

    backend = selected_model_backend(override=model_backend)
    if backend == "tabpfn":
        if _report_training_mode:
            print("training_mode: all_at_once", flush=True)
        model, auc, _ = fit_tabpfn_classifier(
            train_batch_factory,
            feature_columns,
            target_column,
            val_inputs,
            val_target,
            cat_features=cat_features,
        )
        return model, auc

    if train_all_at_once_enabled(train_all_at_once):
        if _report_training_mode:
            print("training_mode: all_at_once", flush=True)
        train_inputs, train_target = _materialize_tabpfn_training_data(
            train_batch_factory,
            feature_columns,
            target_column,
        )
        model, _, auc = fit_tuned_classifier(
            train_inputs,
            train_target,
            val_inputs,
            val_target,
            cat_features=cat_features,
            auto_class_weights=auto_class_weights,
            configs=(config,),
            model_backend=backend,
        )
        return model, auc

    if _report_training_mode:
        print("training_mode: incremental", flush=True)

    iterations = max(1, ceil(int(config["iterations"]) / max(1, batch_count)))
    params = _incremental_model_params(config, "Logloss", iterations)
    if auto_class_weights is not None:
        params["auto_class_weights"] = auto_class_weights

    model: CatBoostClassifier | None = None
    for batch in train_batch_factory():
        target = batch[target_column]
        if target.nunique(dropna=True) < 2:
            continue
        inputs = batch.reindex(columns=list(feature_columns), fill_value=0).fillna(0)
        next_model = CatBoostClassifier(**params)
        next_model.fit(
            inputs,
            target,
            cat_features=cat_features,
            init_model=model,
            use_best_model=False,
        )
        model = next_model

    if model is None:
        raise RuntimeError("Incremental classifier received no batch with two target classes")
    predictions = model.predict_proba(val_inputs)[:, 1]
    return model, float(roc_auc_score(val_target.to_numpy(dtype="float64"), predictions))


def fit_incremental_regressor(
    train_batch_factory: Callable[[], Iterator[pd.DataFrame]],
    feature_columns: Sequence[str],
    target_column: str,
    val_inputs: pd.DataFrame,
    val_target: pd.Series,
    *,
    batch_count: int,
    config: dict[str, Any],
    cat_features: Sequence[int] | None = None,
    model_backend: str | None = None,
    train_all_at_once: bool | None = None,
    _report_training_mode: bool = True,
) -> tuple[Any, float]:
    """Fit one regressor jointly or by continuing across disk-backed batches."""

    backend = selected_model_backend(override=model_backend)
    if backend == "tabpfn":
        if _report_training_mode:
            print("training_mode: all_at_once", flush=True)
        model, mae, _ = fit_tabpfn_regressor(
            train_batch_factory,
            feature_columns,
            target_column,
            val_inputs,
            val_target,
        )
        return model, mae

    if train_all_at_once_enabled(train_all_at_once):
        if _report_training_mode:
            print("training_mode: all_at_once", flush=True)
        train_inputs, train_target = _materialize_tabpfn_training_data(
            train_batch_factory,
            feature_columns,
            target_column,
        )
        model, _, mae = fit_tuned_regressor(
            train_inputs,
            train_target,
            val_inputs,
            val_target,
            cat_features=cat_features,
            configs=(config,),
            model_backend=backend,
        )
        return model, mae

    if _report_training_mode:
        print("training_mode: incremental", flush=True)

    iterations = max(1, ceil(int(config["iterations"]) / max(1, batch_count)))
    params = _incremental_model_params(config, "MAE", iterations)
    model: CatBoostRegressor | None = None
    for batch in train_batch_factory():
        target = batch[target_column]
        # CatBoost cannot fit a regression batch whose target has no
        # variation. Sparse early cutoff frames can legitimately have this
        # shape; skip them and continue the incremental model on the next
        # informative frame.
        if target.nunique(dropna=True) < 2:
            continue
        inputs = batch.reindex(columns=list(feature_columns), fill_value=0).fillna(0)
        next_model = CatBoostRegressor(**params)
        next_model.fit(
            inputs,
            target.astype("float64"),
            cat_features=cat_features,
            init_model=model,
            use_best_model=False,
        )
        model = next_model

    if model is None:
        raise RuntimeError(
            "Incremental regressor received no batch with varying targets"
        )
    predictions = np.asarray(model.predict(val_inputs), dtype="float64")
    return model, float(mean_absolute_error(val_target.to_numpy(dtype="float64"), predictions))


def fit_incremental_cross_entropy_model(
    train_batch_factory: Callable[[], Iterator[pd.DataFrame]],
    feature_columns: Sequence[str],
    target_column: str,
    val_inputs: pd.DataFrame,
    val_target: pd.Series,
    *,
    batch_count: int,
    config: dict[str, Any],
    cat_features: Sequence[int] | None = None,
    train_all_at_once: bool | None = None,
    _report_training_mode: bool = True,
) -> tuple[CatBoostClassifier, float]:
    """Fit one fractional-label probability model jointly or incrementally."""

    if train_all_at_once_enabled(train_all_at_once):
        if _report_training_mode:
            print("probability_training_mode: all_at_once", flush=True)
        train_inputs, train_target = _materialize_tabpfn_training_data(
            train_batch_factory,
            feature_columns,
            target_column,
        )
        model, _, mae = fit_tuned_cross_entropy_model(
            train_inputs,
            train_target,
            val_inputs,
            val_target,
            cat_features=cat_features,
            configs=(config,),
        )
        return model, mae

    if _report_training_mode:
        print("probability_training_mode: incremental", flush=True)

    iterations = max(1, ceil(int(config["iterations"]) / max(1, batch_count)))
    params = _incremental_model_params(config, "CrossEntropy", iterations)
    model: CatBoostClassifier | None = None
    for batch in train_batch_factory():
        target = batch[target_column].astype("float64")
        if target.nunique(dropna=True) < 2:
            continue
        if (
            not np.isfinite(target.to_numpy()).all()
            or ((target < 0.0) | (target > 1.0)).any()
        ):
            raise ValueError(
                "Cross-entropy targets must be finite and lie in [0, 1]"
            )
        inputs = batch.reindex(columns=list(feature_columns), fill_value=0).fillna(0)
        next_model = CatBoostClassifier(**params)
        next_model.fit(
            inputs,
            target,
            cat_features=cat_features,
            init_model=model,
            use_best_model=False,
        )
        model = next_model

    if model is None:
        raise RuntimeError(
            "Incremental cross-entropy model received no batch with varying targets"
        )
    predictions = np.asarray(model.predict_proba(val_inputs)[:, 1], dtype="float64")
    return model, float(
        mean_absolute_error(val_target.to_numpy(dtype="float64"), predictions)
    )


def fit_tuned_classifier_incremental(
    train_batch_factory: Callable[[], Iterator[pd.DataFrame]],
    feature_columns: Sequence[str],
    target_column: str,
    val_inputs: pd.DataFrame,
    val_target: pd.Series,
    *,
    batch_count: int,
    cat_features: Sequence[int] | None = None,
    auto_class_weights: str | None = None,
    configs: Sequence[dict[str, Any]] = CLASSIFIER_CONFIGS,
    model_backend: str | None = None,
    train_all_at_once: bool | None = None,
) -> tuple[Any, dict[str, Any], float]:
    backend = selected_model_backend(override=model_backend)
    if backend == "tabpfn":
        print("training_mode: all_at_once", flush=True)
        model, auc, _ = fit_tabpfn_classifier(
            train_batch_factory,
            feature_columns,
            target_column,
            val_inputs,
            val_target,
            cat_features=cat_features,
        )
        return model, dict(TABPFN_CONFIG), auc

    if train_all_at_once_enabled(train_all_at_once):
        print("training_mode: all_at_once", flush=True)
        train_inputs, train_target = _materialize_tabpfn_training_data(
            train_batch_factory,
            feature_columns,
            target_column,
        )
        return fit_tuned_classifier(
            train_inputs,
            train_target,
            val_inputs,
            val_target,
            cat_features=cat_features,
            auto_class_weights=auto_class_weights,
            configs=configs,
            model_backend=backend,
        )

    print("training_mode: incremental", flush=True)
    best_model: CatBoostClassifier | None = None
    best_config: dict[str, Any] | None = None
    best_auc = float("-inf")
    for config in configs:
        model, auc = fit_incremental_classifier(
            train_batch_factory,
            feature_columns,
            target_column,
            val_inputs,
            val_target,
            batch_count=batch_count,
            config=config,
            cat_features=cat_features,
            auto_class_weights=auto_class_weights,
            model_backend=backend,
            train_all_at_once=False,
            _report_training_mode=False,
        )
        if auc > best_auc:
            best_model, best_config, best_auc = model, config, auc
    if best_model is None or best_config is None:
        raise RuntimeError("Incremental classifier tuning produced no model")
    return best_model, best_config, best_auc


def fit_tuned_regressor_incremental(
    train_batch_factory: Callable[[], Iterator[pd.DataFrame]],
    feature_columns: Sequence[str],
    target_column: str,
    val_inputs: pd.DataFrame,
    val_target: pd.Series,
    *,
    batch_count: int,
    cat_features: Sequence[int] | None = None,
    configs: Sequence[dict[str, Any]] = REGRESSOR_CONFIGS,
    model_backend: str | None = None,
    train_all_at_once: bool | None = None,
) -> tuple[Any, dict[str, Any], float]:
    backend = selected_model_backend(override=model_backend)
    if backend == "tabpfn":
        print("training_mode: all_at_once", flush=True)
        model, mae, _ = fit_tabpfn_regressor(
            train_batch_factory,
            feature_columns,
            target_column,
            val_inputs,
            val_target,
        )
        return model, dict(TABPFN_CONFIG), mae

    if train_all_at_once_enabled(train_all_at_once):
        print("training_mode: all_at_once", flush=True)
        train_inputs, train_target = _materialize_tabpfn_training_data(
            train_batch_factory,
            feature_columns,
            target_column,
        )
        return fit_tuned_regressor(
            train_inputs,
            train_target,
            val_inputs,
            val_target,
            cat_features=cat_features,
            configs=configs,
            model_backend=backend,
        )

    print("training_mode: incremental", flush=True)
    best_model: CatBoostRegressor | None = None
    best_config: dict[str, Any] | None = None
    best_mae = float("inf")
    for config in configs:
        model, mae = fit_incremental_regressor(
            train_batch_factory,
            feature_columns,
            target_column,
            val_inputs,
            val_target,
            batch_count=batch_count,
            config=config,
            cat_features=cat_features,
            model_backend=backend,
            train_all_at_once=False,
            _report_training_mode=False,
        )
        if mae < best_mae:
            best_model, best_config, best_mae = model, config, mae
    if best_model is None or best_config is None:
        raise RuntimeError("Incremental regressor tuning produced no model")
    return best_model, best_config, best_mae


def fit_tuned_cross_entropy_model_incremental(
    train_batch_factory: Callable[[], Iterator[pd.DataFrame]],
    feature_columns: Sequence[str],
    target_column: str,
    val_inputs: pd.DataFrame,
    val_target: pd.Series,
    *,
    batch_count: int,
    cat_features: Sequence[int] | None = None,
    configs: Sequence[dict[str, Any]] = REGRESSOR_CONFIGS,
    train_all_at_once: bool | None = None,
) -> tuple[CatBoostClassifier, dict[str, Any], float]:
    """Tune a fractional-label probability model using validation MAE."""

    if train_all_at_once_enabled(train_all_at_once):
        print("probability_training_mode: all_at_once", flush=True)
        train_inputs, train_target = _materialize_tabpfn_training_data(
            train_batch_factory,
            feature_columns,
            target_column,
        )
        return fit_tuned_cross_entropy_model(
            train_inputs,
            train_target,
            val_inputs,
            val_target,
            cat_features=cat_features,
            configs=configs,
        )

    print("probability_training_mode: incremental", flush=True)
    best_model: CatBoostClassifier | None = None
    best_config: dict[str, Any] | None = None
    best_mae = float("inf")
    for config in configs:
        model, mae = fit_incremental_cross_entropy_model(
            train_batch_factory,
            feature_columns,
            target_column,
            val_inputs,
            val_target,
            batch_count=batch_count,
            config=config,
            cat_features=cat_features,
            train_all_at_once=False,
            _report_training_mode=False,
        )
        if mae < best_mae:
            best_model, best_config, best_mae = model, config, mae
    if best_model is None or best_config is None:
        raise RuntimeError("Incremental cross-entropy tuning produced no model")
    return best_model, best_config, best_mae

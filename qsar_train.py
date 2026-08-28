"""Tune descriptor QSAR regressors and apply the selected model to generated sets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, loguniform, uniform
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real


def _pipeline(model):
    """Use fold-local imputation and scaling for every candidate model."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def _model_spaces(random_state: int):
    return {
        "kernel_ridge": {
            "estimator": _pipeline(KernelRidge()),
            "random": {
                "model__alpha": loguniform(1e-4, 1e2),
                "model__gamma": loguniform(1e-4, 1e1),
                "model__kernel": ["rbf", "laplacian"],
            },
            "bayes": {
                "model__alpha": Real(1e-4, 1e2, prior="log-uniform"),
                "model__gamma": Real(1e-4, 1e1, prior="log-uniform"),
                "model__kernel": Categorical(["rbf", "laplacian"]),
            },
        },
        "random_forest": {
            "estimator": _pipeline(
                RandomForestRegressor(random_state=random_state, n_jobs=1)
            ),
            "random": {
                "model__n_estimators": randint(60, 500),
                "model__max_depth": [None, 5, 10, 15, 20, 30],
                "model__min_samples_split": randint(2, 21),
                "model__min_samples_leaf": randint(1, 13),
                "model__max_features": uniform(0.3, 0.7),
                "model__bootstrap": [True, False],
            },
            "bayes": {
                "model__n_estimators": Integer(60, 500),
                "model__max_depth": Categorical([None, 5, 10, 15, 20, 30]),
                "model__min_samples_split": Integer(2, 20),
                "model__min_samples_leaf": Integer(1, 12),
                "model__max_features": Real(0.3, 1.0),
                "model__bootstrap": Categorical([True, False]),
            },
        },
        "gradient_boosting": {
            "estimator": _pipeline(
                GradientBoostingRegressor(random_state=random_state)
            ),
            "random": {
                "model__n_estimators": randint(50, 501),
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__max_depth": randint(1, 6),
                "model__min_samples_leaf": randint(1, 16),
                "model__subsample": uniform(0.6, 0.4),
                "model__max_features": [None, "sqrt", 0.5, 0.75, 1.0],
            },
            "bayes": {
                "model__n_estimators": Integer(50, 500),
                "model__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                "model__max_depth": Integer(1, 5),
                "model__min_samples_leaf": Integer(1, 15),
                "model__subsample": Real(0.6, 1.0),
                "model__max_features": Categorical([None, "sqrt", 0.5, 0.75, 1.0]),
            },
        },
    }


def _metrics(y_true, y_pred):
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _predict_in_batches(model, features, batch_size: int = 5_000):
    predictions = []
    for start in range(0, len(features), batch_size):
        predictions.append(model.predict(features.iloc[start : start + batch_size]))
    return np.concatenate(predictions)


def _descriptor_ad_for_queries(
    fitted_pipeline,
    X_train,
    X_query,
    k: int = 5,
    quantile: float = 0.95,
    batch_size: int = 5_000,
):
    """Compute descriptor-only AD scores in the selected pipeline's scaled space."""
    transform = fitted_pipeline[:-1]
    train_scaled = transform.transform(X_train)
    neighbor_model = NearestNeighbors(n_neighbors=k, metric="euclidean")
    neighbor_model.fit(train_scaled)

    training_distances, _ = neighbor_model.kneighbors(
        train_scaled,
        n_neighbors=k + 1,
    )
    threshold = float(np.quantile(training_distances[:, 1:].mean(axis=1), quantile))

    query_scores = []
    for start in range(0, len(X_query), batch_size):
        query_scaled = transform.transform(X_query.iloc[start : start + batch_size])
        distances, _ = neighbor_model.kneighbors(query_scaled, n_neighbors=k)
        query_scores.append(distances.mean(axis=1))
    scores = np.concatenate(query_scores)
    return scores, scores <= threshold, threshold


@dataclass
class TrainingResult:
    """Fitted searches and the model selected by scaffold-grouped CV."""

    searches: dict[str, Any]
    leaderboard: pd.DataFrame
    best_name: str
    best_model: Pipeline
    feature_columns: list[str]


def train_and_select_best_model(
    X: pd.DataFrame,
    y: Sequence[float],
    groups: Sequence[object],
    *,
    random_state: int = 42,
    n_iter: int = 40,
    n_splits: int = 5,
    n_jobs: int = -1,
) -> TrainingResult:
    """Tune all model families and select the lowest scaffold-CV RMSE.

    This training-only API is intended for notebooks that evaluate the held-out
    test set and apply the selected model to generated chemistry themselves.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if X.empty:
        raise ValueError("Training features are empty")
    if not X.columns.is_unique:
        raise ValueError("Training feature names must be unique")

    y_array = np.asarray(y, dtype=float).reshape(-1)
    group_array = np.asarray(groups).reshape(-1)
    if len(X) != len(y_array) or len(X) != len(group_array):
        raise ValueError("X, y, and groups must contain the same number of rows")
    if not np.isfinite(y_array).all():
        raise ValueError("Training targets contain missing or non-finite values")
    if pd.isna(group_array).any():
        raise ValueError("Training scaffold groups contain missing values")
    unique_group_count = len(np.unique(group_array))
    cv_folds = min(int(n_splits), unique_group_count)
    if cv_folds < 2:
        raise ValueError("At least two scaffold groups are required")
    if int(n_iter) < 1:
        raise ValueError("n_iter must be at least 1")

    cv = GroupKFold(n_splits=cv_folds)
    searches: dict[str, Any] = {}
    leaderboard_rows = []

    for model_name, specification in _model_spaces(int(random_state)).items():
        model_searches = {
            f"{model_name}/random": RandomizedSearchCV(
                estimator=specification["estimator"],
                param_distributions=specification["random"],
                n_iter=int(n_iter),
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=int(n_jobs),
                refit=True,
                random_state=int(random_state),
                error_score="raise",
            ),
            f"{model_name}/bayesian": BayesSearchCV(
                estimator=specification["estimator"],
                search_spaces=specification["bayes"],
                n_iter=int(n_iter),
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=int(n_jobs),
                refit=True,
                random_state=int(random_state),
                error_score="raise",
            ),
        }
        for search_name, search in model_searches.items():
            print(f"Fitting {search_name} with {cv_folds}-fold scaffold CV")
            search.fit(X, y_array, groups=group_array)
            searches[search_name] = search
            leaderboard_rows.append(
                {
                    "model": search_name,
                    "scaffold_cv_rmse": float(-search.best_score_),
                    "best_parameters": dict(search.best_params_),
                }
            )

    leaderboard = (
        pd.DataFrame(leaderboard_rows)
        .sort_values("scaffold_cv_rmse", ascending=True)
        .reset_index(drop=True)
    )
    best_name = str(leaderboard.loc[0, "model"])
    return TrainingResult(
        searches=searches,
        leaderboard=leaderboard,
        best_name=best_name,
        best_model=searches[best_name].best_estimator_,
        feature_columns=list(X.columns),
    )


def save_selected_model(
    result: TrainingResult,
    path: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Serialize the fitted model together with its feature contract."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": result.best_model,
        "model_name": result.best_name,
        "feature_columns": result.feature_columns,
        "scaffold_cv_leaderboard": result.leaderboard.to_dict(orient="records"),
        "metadata": dict(metadata or {}),
    }
    joblib.dump(artifact, output_path)
    return output_path


def run_qsar_workflow(
    chembl_train: pd.DataFrame,
    chembl_validation: pd.DataFrame,
    chembl_test: pd.DataFrame,
    generated: pd.DataFrame,
    descriptor_columns: list[str],
    output_dir: str | Path = ".",
    random_iterations: int = 12,
    bayes_iterations: int = 18,
    cv_folds: int = 3,
    random_state: int = 42,
):
    """Tune three regressors, select on validation, test once, and predict generated sets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_train_columns = set(descriptor_columns) | {"pIC50", "scaffold"}
    missing = required_train_columns - set(chembl_train.columns)
    if missing:
        raise KeyError(f"Training data are missing columns: {sorted(missing)}")

    X_train = chembl_train[descriptor_columns].copy()
    y_train = chembl_train["pIC50"].to_numpy(dtype=float)
    groups_train = chembl_train["scaffold"].to_numpy()
    X_validation = chembl_validation[descriptor_columns].copy()
    y_validation = chembl_validation["pIC50"].to_numpy(dtype=float)
    X_test = chembl_test[descriptor_columns].copy()
    y_test = chembl_test["pIC50"].to_numpy(dtype=float)

    if X_train.columns.tolist() != list(descriptor_columns):
        raise RuntimeError("QSAR features differ from descriptor_columns")

    cv = GroupKFold(n_splits=cv_folds)
    model_spaces = _model_spaces(random_state)
    search_rows = []
    model_candidates = {}

    for model_name, specification in model_spaces.items():
        print(f"\n=== {model_name}: random search ===")
        random_search = RandomizedSearchCV(
            estimator=specification["estimator"],
            param_distributions=specification["random"],
            n_iter=random_iterations,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
            random_state=random_state,
            verbose=1,
            error_score="raise",
        )
        random_search.fit(X_train, y_train, groups=groups_train)

        print(f"=== {model_name}: Bayesian search ===")
        bayes_search = BayesSearchCV(
            estimator=specification["estimator"],
            search_spaces=specification["bayes"],
            n_iter=bayes_iterations,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
            random_state=random_state,
            verbose=1,
            error_score="raise",
        )
        bayes_search.fit(X_train, y_train, groups=groups_train)

        searches = {"random": random_search, "bayesian": bayes_search}
        for search_name, search in searches.items():
            search_rows.append(
                {
                    "model": model_name,
                    "search": search_name,
                    "cv_rmse": float(-search.best_score_),
                    "best_params": json.dumps(dict(search.best_params_), default=str),
                }
            )

        # Random and Bayesian searches cover the same broad domain. Advance the
        # one with the better grouped-CV RMSE for this model family.
        winning_search_name, winning_search = max(
            searches.items(),
            key=lambda item: item[1].best_score_,
        )
        estimator = winning_search.best_estimator_
        validation_predictions = estimator.predict(X_validation)
        validation_metrics = _metrics(y_validation, validation_predictions)
        model_candidates[model_name] = {
            "estimator": estimator,
            "search": winning_search_name,
            "cv_rmse": float(-winning_search.best_score_),
            "validation": validation_metrics,
            "params": dict(winning_search.best_params_),
        }
        print(
            f"{model_name}: selected {winning_search_name}; "
            f"CV RMSE={-winning_search.best_score_:.4f}; "
            f"validation RMSE={validation_metrics['rmse']:.4f}"
        )

    best_model_name, best_candidate = min(
        model_candidates.items(),
        key=lambda item: item[1]["validation"]["rmse"],
    )
    best_model = best_candidate["estimator"]

    test_predictions = best_model.predict(X_test)
    test_metrics = _metrics(y_test, test_predictions)
    best_candidate["test"] = test_metrics

    metrics_rows = []
    for model_name, candidate in model_candidates.items():
        metrics_rows.append(
            {
                "model": model_name,
                "selected_search": candidate["search"],
                "cv_rmse": candidate["cv_rmse"],
                "validation_rmse": candidate["validation"]["rmse"],
                "validation_mae": candidate["validation"]["mae"],
                "validation_r2": candidate["validation"]["r2"],
                "selected_as_final": model_name == best_model_name,
            }
        )
    metrics = pd.DataFrame(metrics_rows).sort_values("validation_rmse")
    metrics.loc[metrics["selected_as_final"], "test_rmse"] = test_metrics["rmse"]
    metrics.loc[metrics["selected_as_final"], "test_mae"] = test_metrics["mae"]
    metrics.loc[metrics["selected_as_final"], "test_r2"] = test_metrics["r2"]

    search_results = pd.DataFrame(search_rows).sort_values(["model", "cv_rmse"])
    search_results.to_csv(output_dir / "qsar_hyperparameter_search_results.csv", index=False)
    metrics.to_csv(output_dir / "qsar_model_metrics.csv", index=False)
    joblib.dump(best_model, output_dir / "best_qsar_model.joblib")

    metadata = {
        "selected_model": best_model_name,
        "selected_search": best_candidate["search"],
        "descriptor_columns": list(descriptor_columns),
        "best_params": best_candidate["params"],
        "cv_rmse": best_candidate["cv_rmse"],
        "validation_metrics": best_candidate["validation"],
        "test_metrics": test_metrics,
        "random_iterations": random_iterations,
        "bayes_iterations": bayes_iterations,
        "cv_folds": cv_folds,
    }
    with (output_dir / "best_qsar_model_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, default=str)

    X_generated = generated[descriptor_columns].copy()
    generated_predictions = _predict_in_batches(best_model, X_generated)
    ad_distance, inside_ad, ad_threshold = _descriptor_ad_for_queries(
        best_model,
        X_train,
        X_generated,
    )

    metadata_columns = [
        column
        for column in ("SMILES", "run", "generator", "dataset")
        if column in generated.columns
    ]
    generated_qsar = generated[metadata_columns].copy()
    generated_qsar["predicted_pIC50"] = generated_predictions
    generated_qsar["descriptor_knn_distance"] = ad_distance
    generated_qsar["inside_descriptor_ad"] = inside_ad
    generated_qsar["selected_model"] = best_model_name
    generated_qsar.to_csv(output_dir / "generated_qsar_predictions.csv", index=False)

    summary_group = "dataset" if "dataset" in generated_qsar else "generator"
    generated_summary = (
        generated_qsar.groupby(summary_group)
        .agg(
            molecules=("predicted_pIC50", "size"),
            predicted_pIC50_mean=("predicted_pIC50", "mean"),
            predicted_pIC50_median=("predicted_pIC50", "median"),
            predicted_pIC50_std=("predicted_pIC50", "std"),
            inside_ad_fraction=("inside_descriptor_ad", "mean"),
            median_knn_distance=("descriptor_knn_distance", "median"),
        )
        .reset_index()
    )
    generated_summary.to_csv(output_dir / "generated_qsar_summary.csv", index=False)

    print(f"\nSelected model: {best_model_name}")
    print(f"Validation RMSE: {best_candidate['validation']['rmse']:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Generated-set descriptor AD threshold: {ad_threshold:.4f}")

    return {
        "best_model": best_model,
        "best_model_name": best_model_name,
        "metrics": metrics,
        "search_results": search_results,
        "generated_predictions": generated_qsar,
        "generated_summary": generated_summary,
        "metadata": metadata,
    }

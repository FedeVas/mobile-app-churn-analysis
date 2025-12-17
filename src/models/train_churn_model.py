# src/models/train_churn_model.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from catboost import CatBoostClassifier


@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    catboost_iters: int = 400
    catboost_depth: int = 6
    catboost_lr: float = 0.1


def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if "churn_30d" not in df.columns:
        raise ValueError("В данных нет колонке churn_30d")

    y = df["churn_30d"].astype(int)

    drop_cols = ["Идентификатор устройства", "churn_30d", "first_event", "last_event"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols).copy()

    # если кластер есть — делаем его категориальным
    if "cluster_kmeans" in X.columns:
        X["cluster_kmeans"] = X["cluster_kmeans"].astype("category")

    return X, y


def build_preprocess(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ]
    )

    return preprocess, num_cols, cat_cols


def evaluate_binary(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict:
    y_pred = (y_proba >= threshold).astype(int)

    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()
    auc = roc_auc_score(y_true, y_proba)

    return {
        "roc_auc": float(auc),
        "threshold": float(threshold),
        "classification_report": report,
        "confusion_matrix": cm
    }


def train_logreg(
    X_train, y_train, X_test, y_test,
    preprocess: ColumnTransformer
) -> Dict:
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = evaluate_binary(y_test.values, proba, threshold=0.5)

    return {"pipeline": pipe, "metrics": metrics}


def train_catboost_on_preprocessed(
    X_train, y_train, X_test, y_test,
    preprocess: ColumnTransformer,
    cfg: TrainConfig
) -> Dict:
    # обучаем preprocess на train
    X_train_proc = preprocess.fit_transform(X_train)
    X_test_proc = preprocess.transform(X_test)

    # catboost лучше на dense
    if not isinstance(X_train_proc, np.ndarray):
        X_train_proc = X_train_proc.toarray()
        X_test_proc = X_test_proc.toarray()

    model = CatBoostClassifier(
        iterations=cfg.catboost_iters,
        depth=cfg.catboost_depth,
        learning_rate=cfg.catboost_lr,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=cfg.random_state,
        verbose=False
    )

    model.fit(X_train_proc, y_train)
    proba = model.predict_proba(X_test_proc)[:, 1]

    metrics = evaluate_binary(y_test.values, proba, threshold=0.5)

    return {
        "preprocess": preprocess,
        "model": model,
        "metrics": metrics
    }


def save_metrics_json(metrics: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main(
    input_path: str = "data/processed/user_features_with_clusters.parquet",
    out_dir: str = "reports",
) -> None:
    cfg = TrainConfig()

    df = pd.read_parquet(input_path)
    X, y = split_X_y(df)

    preprocess, num_cols, cat_cols = build_preprocess(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y
    )

    # 1) baseline логрег
    logreg_res = train_logreg(X_train, y_train, X_test, y_test, preprocess)

    # 2) catboost
    # (важно: preprocess уже fitted внутри train_catboost_on_preprocessed,
    #  поэтому для честности создадим новый preprocess, чтобы не переиспользовать fitted состояние)
    preprocess2, _, _ = build_preprocess(X)
    cb_res = train_catboost_on_preprocessed(X_train, y_train, X_test, y_test, preprocess2, cfg)

    # сохраняем метрики
    out_dir = Path(out_dir)
    save_metrics_json(logreg_res["metrics"], out_dir / "metrics_logreg.json")
    save_metrics_json(cb_res["metrics"], out_dir / "metrics_catboost.json")

    # печать в консоль (удобно для запуска)
    print("=== Logistic Regression ===")
    print("ROC-AUC:", logreg_res["metrics"]["roc_auc"])
    print("Confusion matrix:", logreg_res["metrics"]["confusion_matrix"])

    print("\n=== CatBoost ===")
    print("ROC-AUC:", cb_res["metrics"]["roc_auc"])
    print("Confusion matrix:", cb_res["metrics"]["confusion_matrix"])


if __name__ == "__main__":
    main()

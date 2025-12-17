# src/models/cluster_users.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


DEFAULT_BASE_COLS = [
    "events_total",
    "sessions_total",
    "active_days",
    "events_per_day",
    "sessions_per_day",
    "unique_screens",
    "unique_functional",
]


@dataclass
class ClusteringArtifacts:
    scaler: StandardScaler
    pca: PCA
    behaviour_cols: List[str]
    log_cols: List[str]
    n_components: int


def select_behaviour_cols(df: pd.DataFrame) -> List[str]:
    """Берём поведенческие признаки: базовые метрики + доли screen_*."""
    screen_cols = [c for c in df.columns if c.startswith("screen_")]
    cols = [c for c in (DEFAULT_BASE_COLS + screen_cols) if c in df.columns]
    if not cols:
        raise ValueError("Не найдено поведенческих фич для кластеризации (нет base_cols и/или screen_*).")
    return cols


def prepare_features_for_clustering(
    user_features: pd.DataFrame,
    behaviour_cols: Optional[List[str]] = None,
    max_pca_components: int = 10,
    random_state: int = 42,
) -> Tuple[np.ndarray, ClusteringArtifacts]:
    """
    Подготовка X для кластеризации:
      - лог-трансформ перекошенных счетчиков
      - fillna
      - StandardScaler
      - PCA до max_pca_components
    """
    df = user_features.copy()

    if behaviour_cols is None:
        behaviour_cols = select_behaviour_cols(df)

    X = df[behaviour_cols].copy()

    # лог-трансформируем перекошенные счетчики
    log_candidates = ["events_total", "sessions_total", "active_days", "events_per_day", "sessions_per_day"]
    log_cols = [c for c in log_candidates if c in X.columns]
    for col in log_cols:
        X[col] = np.log1p(X[col])

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_features = X_scaled.shape[1]
    n_components = min(max_pca_components, n_features)

    pca = PCA(n_components=n_components, random_state=random_state, svd_solver="randomized")
    X_pca = pca.fit_transform(X_scaled)

    artifacts = ClusteringArtifacts(
        scaler=scaler,
        pca=pca,
        behaviour_cols=behaviour_cols,
        log_cols=log_cols,
        n_components=n_components,
    )
    return X_pca, artifacts


def fit_kmeans(
    X_pca: np.ndarray,
    cluster_range: List[int] = [3, 4, 5, 6, 7, 8, 9],
    random_state: int = 42,
    sample_size_for_silhouette: int = 20000,
) -> Tuple[np.ndarray, Dict[int, float], int, KMeans]:
    """
    KMeans + подбор k по silhouette (на сэмпле, чтобы не убить время/память).
    """
    n = X_pca.shape[0]
    sample_size = min(sample_size_for_silhouette, n)

    sil_scores: Dict[int, float] = {}
    best_k = None
    best_score = -1.0
    best_model = None

    for k in cluster_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = model.fit_predict(X_pca)

        score = silhouette_score(
            X_pca, labels,
            sample_size=sample_size,
            random_state=random_state
        )
        sil_scores[k] = float(score)

        if score > best_score:
            best_score = score
            best_k = k
            best_model = model

    assert best_k is not None and best_model is not None
    final_labels = best_model.predict(X_pca)
    return final_labels, sil_scores, best_k, best_model


def fit_dbscan(
    X_pca: np.ndarray,
    eps: float,
    min_samples: int = 20,
    sample_size_for_silhouette: int = 20000,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, float], DBSCAN]:
    """
    DBSCAN на PCA-пространстве. Silhouette считаем только на точках без шума (-1).
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_pca)

    n_noise = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    metrics: Dict[str, float] = {
        "n_clusters": float(n_clusters),
        "noise_fraction": float(n_noise / len(labels)),
        "silhouette_no_noise": float("nan"),
    }

    mask = labels != -1
    if mask.sum() > 1 and n_clusters > 1:
        sample_size = min(sample_size_for_silhouette, int(mask.sum()))
        metrics["silhouette_no_noise"] = float(
            silhouette_score(
                X_pca[mask], labels[mask],
                sample_size=sample_size,
                random_state=random_state
            )
        )

    return labels, metrics, model


def make_cluster_summary(
    df: pd.DataFrame,
    cluster_col: str,
    churn_col: str = "churn_30d",
) -> pd.DataFrame:
    """
    Таблица для отчёта: размер кластера (%) + churn_rate (%), если churn_col есть.
    """
    out = pd.DataFrame()
    out["cluster_size_%"] = df[cluster_col].value_counts(normalize=True).sort_index() * 100

    if churn_col in df.columns:
        out["churn_rate_%"] = df.groupby(cluster_col)[churn_col].mean().sort_index() * 100

    return out.reset_index().rename(columns={"index": cluster_col})


def cluster_users(
    user_features: pd.DataFrame,
    run_kmeans_flag: bool = True,
    run_dbscan_flag: bool = False,
    dbscan_eps: float = 1.5,
    dbscan_min_samples: int = 20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Основная функция: готовит признаки и запускает KMeans и/или DBSCAN.
    Возвращает user_features с колонками:
      - cluster_kmeans (если run_kmeans_flag)
      - cluster_dbscan (если run_dbscan_flag)
    """
    X_pca, prep_artifacts = prepare_features_for_clustering(user_features, random_state=random_state)

    result = user_features.copy()
    artifacts: Dict = {
        "prep": prep_artifacts,
        "pca_explained_variance_ratio": prep_artifacts.pca.explained_variance_ratio_.tolist(),
    }

    if run_kmeans_flag:
        labels_km, sil_scores, best_k, kmeans_model = fit_kmeans(
            X_pca,
            random_state=random_state,
        )
        result["cluster_kmeans"] = labels_km
        artifacts["kmeans"] = {
            "best_k": best_k,
            "silhouette_scores": sil_scores,
            "model": kmeans_model,
        }

    if run_dbscan_flag:
        labels_db, db_metrics, db_model = fit_dbscan(
            X_pca,
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
            random_state=random_state,
        )
        result["cluster_dbscan"] = labels_db
        artifacts["dbscan"] = {
            "metrics": db_metrics,
            "model": db_model,
        }

    return result, artifacts

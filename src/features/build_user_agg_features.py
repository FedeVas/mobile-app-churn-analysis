# src/features/build_user_agg_features.py
import pandas as pd

def build_user_features(events: pd.DataFrame, horizon_days: int = 30) -> pd.DataFrame:
    """
    Строит фичи на уровне устройства и таргет churn_30d.
    Окно наблюдения = весь период минус horizon_days.
    """
    df = events.copy()
    max_date = df['Дата и время события'].max()
    cutoff_date = max_date - pd.Timedelta(days=horizon_days)

    obs = df[df['Дата и время события'] < cutoff_date]
    target_window = df[df['Дата и время события'] >= cutoff_date]

    # базовые агрегаты
    user_agg = obs.groupby('Идентификатор устройства').agg(
        events_total=('Дата и время события', 'size'),
        sessions_total=('session_id', 'nunique'),
        first_event=('Дата и время события', 'min'),
        last_event=('Дата и время события', 'max'),
        unique_screens=('Экран', 'nunique'),
        unique_functional=('Функционал', 'nunique'),
    ).reset_index()

    user_agg['active_days'] = (user_agg['last_event'] - user_agg['first_event']).dt.days + 1
    user_agg['events_per_day'] = user_agg['events_total'] / user_agg['active_days'].clip(lower=1)
    user_agg['sessions_per_day'] = user_agg['sessions_total'] / user_agg['active_days'].clip(lower=1)

    # профиль по экранам (топ-8)
    top_screens = obs['Экран'].value_counts().head(8).index
    screen_pivot = (
        obs[obs['Экран'].isin(top_screens)]
        .groupby(['Идентификатор устройства', 'Экран'])
        .size()
        .unstack(fill_value=0)
    )
    screen_pivot = screen_pivot.div(screen_pivot.sum(axis=1), axis=0).fillna(0)
    screen_pivot = screen_pivot.add_prefix('screen_').reset_index()

    # статические фичи по устройству
    def mode_or_nan(x):
        return x.mode().iloc[0] if not x.mode().empty else None

    user_static = obs.groupby('Идентификатор устройства').agg(
        os=('ОС', mode_or_nan),
        device_type=('Тип устройства', mode_or_nan),
        manufacturer=('Производитель устройства', mode_or_nan),
    ).reset_index()

    # таргет churn
    active_in_target = target_window['Идентификатор устройства'].unique()
    active_in_target = pd.Series(1, index=active_in_target, name='is_active_target')

    target_df = user_agg[['Идентификатор устройства']].merge(
        active_in_target,
        left_on='Идентификатор устройства',
        right_index=True,
        how='left'
    ).fillna(0)

    target_df['is_active_target'] = target_df['is_active_target'].astype(int)
    target_df['churn_30d'] = 1 - target_df['is_active_target']

    # склейка
    user_features = (
        user_agg
        .merge(screen_pivot, on='Идентификатор устройства', how='left')
        .merge(user_static, on='Идентификатор устройства', how='left')
        .merge(target_df[['Идентификатор устройства', 'churn_30d']], on='Идентификатор устройства', how='left')
        .fillna(0)
    )

    return user_features

# src/data/load_events.py
import pandas as pd

def load_raw_events(path: str) -> pd.DataFrame:
    """Читает сырой csv с событиями и приводит дату к datetime."""
    df = pd.read_csv(path)
    df["Дата и время события"] = pd.to_datetime(
        df["Дата и время события"].str.split('[', n=1).str[0],
        utc=True,
        format="ISO8601"
    )
    return df


def add_basic_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет date/hour/weekday, не трогая остальное."""
    df = df.copy()
    df['date'] = df['Дата и время события'].dt.date
    df['hour'] = df['Дата и время события'].dt.hour
    df['weekday'] = df['Дата и время события'].dt.dayofweek
    df['session_id'] = (
        df['Идентификатор устройства'].astype(str)
        + '_'
        + df['Номер сессии в рамках устройства'].astype(str)
    )
    return df

# mobile-app-churn-analysis

Проект по анализу поведения пользователей и прогнозированию оттока в мобильном приложении ЖКХ.

## 1. Структура проекта
```text
mobile-app-churn-analysis/
├─ README.md
├─ requirements.txt
├─ .gitignore
│
├─ data/
│  ├─ raw/          # исходные csv/parquet
│  └─ processed/    # очищенные и агрегированные данные (user-level, features)
│
├─ notebooks/
│  ├─ 01_eda_events.ipynb             # базовый EDA + временные паттерны
│  ├─ 02_user_features_churn_label.ipynb  # фичи по пользователю + целевая переменная "отток"
│  ├─ 03_churn_modeling.ipynb         # модели оттока, метрики, интерпретация
│  └─ 04_behavior_clustering.ipynb    # кластеры поведения + профили пользователей
│
├─ src/
│  ├─ data/
│  │  ├─ load_events.py               # функции чтения исходных данных
│  │  └─ preprocess_events.py         # очистка, нормализация, подготовка событий
│  │
│  ├─ features/
│  │  ├─ build_user_agg_features.py   # агрегаты по пользователю (RFM, частоты событий)
│  │  └─ build_journey_features.py    # фичи по customer-journey (экраны, переходы) (если успеете)
│  │
│  ├─ models/
│  │  ├─ train_churn_model.py         # обучение модели оттока (CatBoost/LightGBM/склад)
│  │  └─ cluster_users.py             # кластеризация пользователей (KMeans и пр.)
│  │
│  └─ viz/
│     ├─ eda_plots.py                 # функции построения графиков EDA
│     └─ journey_maps.py              # графы/диаграммы переходов между экранами
│
├─ reports/
│  ├─ eda_report.md                   # текстовый отчёт EDA + ключевые инсайты
│  └─ figures/                        # сохранённые картинки графиков
│
└─ presentation/
   └─ slides.pptx                     # финальная презентация
```

## 2. Пайплайн

1. `data/raw/датасет_new.csv` — сырой датасет событий (не хранится в репозитории).
2. `notebooks/01_eda_events.ipynb` — EDA по событиям.
3. `notebooks/02_user_features_churn_label.ipynb` — построение фич и метки churn_30d.
4. `notebooks/03_churn_modeling.ipynb` — модели прогнозирования оттока.
5. `notebooks/04_behavior_clustering.ipynb` — кластеризация пользователей по поведению.

## 3. Как запустить

1. Создать виртуальное окружение и установить зависимости:
   ```bash
   pip install -r requirements.txt

2. Положить сырой файл в data/raw/.
3. Запустить ноутбуки в порядке 01 → 02 → 03 / 04.


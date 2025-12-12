```text
mobile-app-churn-analysis/
├─ README.md
├─ requirements.txt
├─ .gitignore
│
├─ data/
│  ├─ raw/          # исходные csv/parquet, НЕ коммитить крупные данные
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
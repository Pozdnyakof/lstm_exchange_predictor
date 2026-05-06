# Дипломная работа: инструмент алгоритмической торговли

Реализация программного прототипа из глав 2-3 ВКР. Гибридная сеть
1D-CNN + LSTM с MC Dropout, мультигоризонтная регрессия нормализованной
лог-доходности, **live-инференс через MOEX ISS** и веб-интерфейс
трейдера на FastAPI + Plotly.

## Архитектура

```
ИССЛЕДОВАТЕЛЬСКИЙ РЕЖИМ                        ПОЛЬЗОВАТЕЛЬСКИЙ РЕЖИМ (LIVE)
─────────────────────────                      ─────────────────────────────
notebooks/training_pipeline.ipynb              FastAPI (inference service)
    ↓ сохраняет                                    ↑ загружает
    checkpoints/model_artifact.pt   ←─────── (Google Drive → local)
              + scaler.json                       ↓ периодически опрашивает
              + meta.json                       MOEX ISS → MC Dropout → JSON
                                                   ↓
                                               Frontend: live-карточки + алерты

ОФЛАЙН-ОТЧЁТ (для §3.4 защиты)
──────────────────────────────
scripts/04_backtest.py
    + страница /report (бэктест aggregate + per-ticker + random monkeys)
```

## Структура

```
graduate_work/
├── data/
│   ├── raw/                           CSV: первичные котировки
│   └── processed/
│       ├── features.parquet           объединённый признаковый фрейм
│       └── runtime/                   артефакты бэктеста для /report
├── checkpoints/                       артефакт-пакет модели
│   ├── model_artifact.pt
│   ├── scaler.json
│   └── meta.json
├── src/graduate_work/
│   ├── config.py                      все гиперпараметры (Data/Model/Training/Trading/Serving)
│   ├── data/                          модуль 1: MOEX ISS, ЦБ РФ, Yahoo Finance
│   ├── features/                      модуль 2: индикаторы, таргеты, скейлер, окна
│   ├── model/                         архитектура 1D-CNN + LSTM, MC Dropout
│   ├── training/                      обучение и MC-инференс
│   ├── strategy/                      двухступенчатый фильтр сигналов
│   ├── backtest/                      движок, метрики, per-ticker, random monkeys
│   ├── serving/                       артефакт-пакет, live-features, scheduler
│   └── web/                           FastAPI + Plotly + Jinja
│       ├── app.py                     wiring + lifespan
│       ├── routes_live.py             /api/live/*
│       ├── routes_report.py           /api/report/*
│       ├── deps.py                    общие зависимости
│       └── templates/                 _base.html, live.html, report.html
├── notebooks/
│   └── training_pipeline.ipynb        §3.2 + §3.4: обучение → save → бэктест → random monkeys
├── scripts/
│   ├── 01_download_data.py            скачать сырые данные
│   ├── 02_build_features.py           собрать признаковую таблицу
│   ├── 03_train_model.py              headless-альтернатива блокноту
│   ├── 04_backtest.py                 офлайн-отчёт для /report
│   ├── 05_run_server.py               запустить веб-интерфейс
│   └── 06_smoke_live.py               проверка live-инференса без UI
└── tests/                             pytest-сьют (features, model, strategy, backtest, serving)
```

## Установка

Требуется Python 3.11 или 3.12.

```bash
cd graduate_work
poetry install
# либо
pip install -e .
```

## Полный цикл

```bash
# 1) Загрузка данных и обучение
poetry run python scripts/01_download_data.py    # ~5 минут
poetry run python scripts/02_build_features.py
poetry run jupyter notebook notebooks/training_pipeline.ipynb
# либо headless: poetry run python scripts/03_train_model.py

# 2) Офлайн-отчёт (для страницы /report)
poetry run python scripts/04_backtest.py

# 3) Веб-интерфейс
poetry run python scripts/05_run_server.py       # http://127.0.0.1:8000
```

После прохождения блокнота в `checkpoints/` появятся три файла
артефакт-пакета (`model_artifact.pt`, `scaler.json`, `meta.json`).
Бэкенд на старте загружает их и поднимает `RefreshScheduler`,
обновляющий прогнозы каждые 15 минут (см. `ServingConfig`).

## API

### Live (главная страница `/`)
| Метод | Путь | Назначение |
|---|---|---|
| GET | `/api/live/predictions` | прогнозы по всем тикерам, отсортированные по `expected_return` |
| GET | `/api/live/alerts` | только активные алерты (BUY с достаточным SNR) |
| GET | `/api/live/{ticker}` | детальный прогноз одного тикера |
| POST | `/api/live/refresh` | принудительное обновление кэша |

### Report (страница `/report`, §3.4)
| Метод | Путь | Назначение |
|---|---|---|
| GET | `/api/report/backtest` | aggregate equity curve + метрики |
| GET | `/api/report/per_ticker` | per-ticker таблица + bar-chart |
| GET | `/api/report/random` | гистограмма random monkeys + 3σ-проверка |

## Логика алертов

```
alert = (mean ≥ min_expected_return)
      AND (std ≤ max_uncertainty)
      AND (mean / std ≥ alert_min_strength)
```

Это естественное расширение двухступенчатого фильтра §2.2: первые два
условия - тот же BUY-фильтр, что в офлайн-стратегии; третье добавляет
порог signal-to-noise, фильтруя «слабые» уверенные прогнозы.

## Тесты

```bash
poetry run pytest -q
```

20 тестов, ~6 секунд. Не требуют сети: HTTP-клиент MOEX замокирован.

| Файл | Что проверяет |
|---|---|
| `test_features.py` | индикаторы, нормализация, скейлер с fit-only-on-train |
| `test_model.py` | размерность тензора, ненулевая дисперсия MC Dropout, toggle MC-режима |
| `test_strategy.py` | HOLD при всех отрицательных, отсечение по std, top-K cap |
| `test_backtest.py` | сделки, отсутствие сигналов, random monkeys, per-ticker |
| `test_serving.py` | сохранение/загрузка артефакта, live-features с моком ISS, инференс |

## Соответствие главам ВКР

| Глава | Где в коде |
|---|---|
| §1.2 - норм. лог-доходность | `features/targets.py` |
| §1.2 - MC Dropout | `model/mc_dropout.py`, `training/inference.py` |
| §1.3 - random monkeys, 3σ | `backtest/random_portfolios.py` |
| §2.1 - выбор 1D-CNN + LSTM | `model/conv_lstm.py` |
| §2.2 - 5 модулей | пакеты `data`, `features`, `model`+`strategy`, `backtest`, `web`+`serving` |
| §2.3 - технологический стек | `pyproject.toml` |
| §3.1 - получение данных через API | `data/`, `serving/live_features.py` |
| §3.2 - обучение модели | `notebooks/training_pipeline.ipynb` |
| §3.3 - веб-интерфейс трейдера | `web/templates/live.html`, `routes_live.py` |
| §3.4 - тестирование на исторических данных | `notebooks/training_pipeline.ipynb` (ячейки 7-9), `scripts/04_backtest.py`, `web/templates/report.html` |

## Замечания

- Параметры `commission_rate` и `slippage_rate` в `TradingConfig`
  зафиксированы на нуле для чистого сравнения с random monkeys.
  Инфраструктура учёта в `engine.py` и `random_portfolios.py` оставлена
  нетронутой - это соответствует тексту §1.3 и §2.2 ВКР, требующего
  «учёт реальных брокерских комиссий и проскальзывания».
- Список тикеров (30 наиболее ликвидных бумаг МосБиржи) задан в
  `DataConfig.tickers`; добавление новых - одна строка.
- В live-режиме макроряды (ЦБ, Brent) не подгружаются; соответствующие
  колонки заполняются нулями после нормализации (что соответствует
  «среднему наблюдённому уровню»). При желании можно добавить
  отдельный live-fetch для них в `LiveFeatureBuilder._fetch_and_build`.

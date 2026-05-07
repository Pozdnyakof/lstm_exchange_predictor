"""Конфигурация инструмента прогнозирования.

Все параметры собраны в иерархию dataclass-ов; значения по умолчанию
соответствуют решениям, обоснованным в главах 1-2 ВКР.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# config.py is at <project>/src/graduate_work/config.py -> parents[2] = <project>
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Файловые пути проекта."""

    project_root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    checkpoints: Path = PROJECT_ROOT / "checkpoints"

    def ensure(self) -> None:
        for p in (self.data_raw, self.data_processed, self.checkpoints):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DataConfig:
    """Параметры сбора и подготовки данных (модули 1-2)."""

    # 12 тикеров, отобранные по результатам предыдущего per-ticker бэктеста
    # (ранжирование по комбинации Sharpe / total_return / win_rate).
    # CSV всех 30 тикеров остаются на диске и Drive — для перерасчёта
    # достаточно поменять этот кортеж.
    tickers: tuple[str, ...] = (
        "GMKN",
        "PLZL",
    )
    start_date: str = "2020-01-01"
    end_date: str = "2026-01-01"
    # MOEX ISS поддерживает интервалы 1, 10, 60, 24, 7, 31. Качаем 1-мин и
    # ресэмплируем до bar_minutes - так получаем любой целевой таймфрейм
    # без отдельного API.
    moex_interval: int = 1
    # Целевой таймфрейм после ресэмпла. 5 минут даёт богатую интрадей-сетку
    # для коротких горизонтов прогноза (5/15/30/60 мин). Сырые данные на
    # диске - 1-минутные, ресэмплинг происходит при построении признаков
    # и в live-режиме.
    bar_minutes: int = 5
    # Батчирование загрузки: качаем по download_batch_months месяцев за раз,
    # сохраняем CSV после каждого батча, чтобы при сбое MOEX-сессии или
    # rate-limit не потерять прогресс. На каждый батч - download_batch_retries
    # попыток с экспоненциальным backoff.
    download_batch_months: int = 3
    download_batch_retries: int = 4
    download_batch_backoff_sec: float = 5.0
    # Параллельная загрузка тикеров. 5 - разумный компромисс: I/O-bound,
    # MOEX ISS обычно держит, ретраи прикрывают временные блокировки.
    # Если MOEX начал агрессивно резать - снижайте до 2-3.
    download_workers: int = 5
    # MOEX основная сессия: 10:00 - 18:45 МСК = 07:00 - 15:45 UTC.
    session_start_utc: str = "07:00"
    session_end_utc: str = "15:45"
    base_index: str = "IMOEX"
    extra_indexes: tuple[str, ...] = ("RGBI", "RTSI")
    brent_symbol: str = "BZ=F"  # Yahoo Finance тикер Brent
    cbr_currencies: tuple[str, ...] = ("USD", "EUR")
    # Горизонты прогноза в барах (5-минутных):
    # 6=30мин, 12=1ч, 24=2ч, 48=4ч.
    # Длинные горизонты компенсируют комиссии/проскальзывание (round-trip
    # ≈0.10%): на 4-часовой дистанции реалистичные движения легко
    # перекрывают costs, что улучшает баланс классов в cost-aware метках.
    horizons: tuple[int, ...] = (6, 12, 24, 48)
    # 32 часа контекста (~3.6 сессии). Соотношение look-back:forecast = 8:1
    # под максимальный горизонт 4ч — соответствует R-0023 baseline
    # (15-мин бары, seq_len=128, max_h=16 бар: 128/16=8). На 5-мин барах
    # тот же 8:1 даёт 384 бара контекста.
    window_size: int = 384
    # T2.1: per-ticker dummies как exogenous-фичи.
    use_ticker_dummies: bool = True

    # === РЕЖИМ ОБУЧЕНИЯ ===
    # "regression"     — model выдаёт нормализованную лог-доходность,
    #                    Huber-loss, фильтр по mean (предыдущий путь).
    # "classification" — бинарный таргет "прибыль ≥ 0 после костов",
    #                    BCE-loss, фильтр по вероятности (Bayes-порог).
    # При смене ОБЯЗАТЕЛЬНО переобучить модель.
    mode: str = "classification"
    # Сглаживание меток (label smoothing) для классификации:
    # 0.0 → жёсткие {0, 1}; 0.05 → {0.05, 0.95}.
    label_smoothing: float = 0.0
    # Если True - голова обучается предсказывать выгодность ШОРТА (lr<0)
    # вместо лонга. По умолчанию False (стандартный лонг-фокус).
    swap_long_short_labels: bool = False
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # tail = 1 - train_ratio - val_ratio

    @property
    def bar_timedelta(self) -> pd.Timedelta:
        """Length of one bar as a pandas Timedelta."""
        return pd.Timedelta(minutes=self.bar_minutes)

    @property
    def bars_per_year(self) -> float:
        """Bars in a trading year (252 days * bars/session)."""
        start = pd.Timestamp(self.session_start_utc)
        end = pd.Timestamp(self.session_end_utc)
        session_minutes = int((end - start).total_seconds() / 60) + 1
        bars_per_session = max(1, session_minutes // self.bar_minutes)
        return float(252 * bars_per_session)


@dataclass(frozen=True)
class ModelConfig:
    """Гиперпараметры архитектуры.

    ``architecture`` выбирает фактическую сеть:
    - ``"timexer"`` — Transformer-baseline из исследовательского
      журнала (R-0023, R09.M). Используется по умолчанию.
    - ``"conv_lstm"`` — гибридная 1D-CNN + LSTM, исходная архитектура §2.2.
    """

    architecture: str = "timexer"

    # === Общие параметры ===
    fc_hidden: int = 64
    dropout: float = 0.3
    # Reversible Instance Normalization (Kim et al. 2022) поверх входа.
    use_revin: bool = True
    revin_affine: bool = True

    # === ConvLSTM-параметры (architecture="conv_lstm") ===
    conv_channels: int = 64
    conv_kernel: int = 5
    lstm_hidden: int = 128
    lstm_layers: int = 2

    # === TimeXer-параметры (architecture="timexer") ===
    # Значения соответствуют R-0023 baseline; seq_len выровнен под
    # window_size=384 (32 часа): patch_len=48, stride=24 -> 15 патчей,
    # та же patch-сетка, что у research (15 патчей при seq=128/p=16/s=8).
    timexer_d_model: int = 128
    timexer_n_layers: int = 3
    timexer_n_heads: int = 8
    timexer_d_ff: int = 256
    timexer_patch_len: int = 48
    timexer_stride: int = 24
    timexer_seq_len: int = 384
    timexer_dropout: float = 0.3
    # n_exo=0: все каналы эндогенные (в т.ч. ticker dummies). Если
    # отделять exo (например, индексные returns) - указывается их
    # количество, и они попадают в последние n_exo колонок входа.
    timexer_n_exo: int = 0


@dataclass(frozen=True)
class TrainingConfig:
    """Параметры обучения.

    Параметры по умолчанию рассчитаны на L4 (40 GB VRAM) / Colab Pro+.
    На T4 (16 GB VRAM) рекомендуется снизить batch_size до 256-512.
    """

    batch_size: int = 2048  # L4 легко выдерживает 2-4K при window=30
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 6
    grad_clip: float = 1.0
    seed: int = 42
    # 100 проходов вместо 50 - точнее оценка эпистемической
    # неопределённости, инференс остаётся быстрым.
    mc_passes: int = 100
    # Stochastic Weight Averaging (Izmailov et al. 2018) - усреднение
    # весов после swa_start_frac × epochs. Даёт более плоский минимум
    # loss landscape и снижает дисперсию по сидам.
    use_swa: bool = True
    swa_start_frac: float = 0.5
    swa_lr: float = 5e-4
    # T1.2: Huber-δ автоматически вычисляется из распределения таргетов
    # (≈ 2 × median(|y_train|)). Защищает от ошибки когда δ=1.0 слишком
    # большой для нормализованных лог-доходностей масштаба ~1e-3.
    huber_delta_auto: bool = True
    # T2.2: AdamW + CosineAnnealing вместо плоского Adam.
    optimizer: str = "adamw"  # adam | adamw
    scheduler: str = "cosine"  # none | cosine


@dataclass(frozen=True)
class TradingConfig:
    """Торговые ограничения и пороги фильтрации."""

    initial_capital: float = 1_000_000.0
    # Реалистичные транзакционные издержки на MOEX:
    # - брокерская комиссия ~0.03% за сторону (типично для розничных тарифов)
    # - проскальзывание ~0.02% (либералиновая для высоколиквидных топов)
    # Round-trip ≈ 0.10%, что соответствует §2.2 ВКР про «эмулирует
    # исполнение заявки с учётом реальных брокерских комиссий и
    # проскальзывания». Random monkeys учитывают те же издержки симметрично.
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0002
    max_positions: int = 5  # сколько активов держим одновременно
    # === REGRESSION (старый путь) ===
    min_expected_return: float = 0.0005  # порог mean-прогноза
    max_uncertainty: float = 0.02  # порог std-прогноза
    horizon_argmax_correction: float = 1.5

    # === CLASSIFICATION ===
    # Базовый порог вероятности для BUY-сигнала. Финальный порог
    # вычисляется из Bayes-формулы (cost / (cost + gain)) на val,
    # это значение - fallback при неудаче калибровки.
    probability_threshold: float = 0.55
    # Максимальный std MC-прогноза вероятности. Для классификации
    # используется meaningful range [0, 1], 0.25 - умеренная отсечка.
    max_probability_std: float = 0.25
    # Šidák-коррекция при выборе argmax-горизонта (исследование R-0007):
    # T_eff = T^(1/N_horizons). "none" | "sidak" | "bonferroni"
    selection_correction: str = "sidak"

    # === LOSS ===
    # "bce"   — обычный binary cross-entropy (дефолт для классификации)
    # "focal" — focal-loss; полезен при дисбалансе классов
    loss_objective: str = "bce"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    # Случайные портфели:
    n_random_portfolios: int = 1000
    sigma_threshold: float = 3.0


@dataclass(frozen=True)
class ServingConfig:
    """Параметры live-инференса (модуль 5)."""

    refresh_interval_sec: int = 900  # как часто фоновый scheduler обновляет кэш
    live_buffer_days: int = (
        60  # сколько дополнительных дней качать (запас под индикаторы)
    )
    moex_request_pause: float = 0.2  # троттлинг между запросами к MOEX ISS
    alert_min_strength: float = (
        2.0  # min mean / std (signal-to-noise) для алерта
    )
    cache_ttl_sec: int = 300  # TTL кэша price-окон в памяти


@dataclass
class ExperimentConfig:
    """Корневой конфиг."""

    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)


def default_config() -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.paths.ensure()
    return cfg

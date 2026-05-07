"""Конфигурация инструмента прогнозирования.

Все параметры собраны в иерархию dataclass-ов; значения по умолчанию
соответствуют решениям, обоснованным в главах 1-2 ВКР.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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

    tickers: tuple[str, ...] = (
        # Голубые фишки IMOEX
        "SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK",
        "MTSS", "MGNT", "PLZL", "TATN", "CHMF", "ALRS",
        "SNGS", "YDEX",
        # Расширение - ликвидные бумаги с историей с 2022-09
        "VTBR", "MOEX", "PIKK", "PHOR", "AFKS", "HYDR",
        "IRAO", "RUAL", "NLMK", "MAGN", "AFLT", "SIBN",
        "RTKM", "MTLR", "FLOT", "SMLT",
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
    # Горизонты прогноза в барах (5-минутных): 1=5мин, 3=15мин, 6=30мин, 12=1ч.
    horizons: tuple[int, ...] = (1, 3, 6, 12)
    window_size: int = 48           # 4 часа контекста ~ половина торг. сессии
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # tail = 1 - train_ratio - val_ratio


@dataclass(frozen=True)
class ModelConfig:
    """Гиперпараметры гибридной сети 1D-CNN + LSTM."""

    conv_channels: int = 64
    conv_kernel: int = 5
    lstm_hidden: int = 128
    lstm_layers: int = 2
    fc_hidden: int = 64
    dropout: float = 0.3
    # Reversible Instance Normalization (Kim et al. 2022) поверх входа.
    # Адаптивная per-instance нормализация дополняет глобальный
    # StandardScaler и помогает при distribution shift между периодами.
    use_revin: bool = True
    revin_affine: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    """Параметры обучения.

    Параметры по умолчанию рассчитаны на L4 (40 GB VRAM) / Colab Pro+.
    На T4 (16 GB VRAM) рекомендуется снизить batch_size до 256-512.
    """

    batch_size: int = 2048           # L4 легко выдерживает 2-4K при window=30
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


@dataclass(frozen=True)
class TradingConfig:
    """Торговые ограничения и пороги фильтрации."""

    initial_capital: float = 1_000_000.0
    # Параметры транзакционных издержек оставлены в инфраструктуре
    # учёта (engine, random_portfolios), но в текущем эксперименте
    # зафиксированы на нуле для чистого сравнения с random monkeys.
    commission_rate: float = 0.0
    slippage_rate: float = 0.0
    max_positions: int = 5            # сколько активов держим одновременно
    min_expected_return: float = 0.0005   # порог по среднему MC прогнозу
    max_uncertainty: float = 0.02         # порог по std MC прогноза
    # Случайные портфели:
    n_random_portfolios: int = 1000
    sigma_threshold: float = 3.0


@dataclass(frozen=True)
class ServingConfig:
    """Параметры live-инференса (модуль 5)."""

    refresh_interval_sec: int = 900       # как часто фоновый scheduler обновляет кэш
    live_buffer_days: int = 60            # сколько дополнительных дней качать (запас под индикаторы)
    moex_request_pause: float = 0.2       # троттлинг между запросами к MOEX ISS
    alert_min_strength: float = 2.0       # min mean / std (signal-to-noise) для алерта
    cache_ttl_sec: int = 300              # TTL кэша price-окон в памяти


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

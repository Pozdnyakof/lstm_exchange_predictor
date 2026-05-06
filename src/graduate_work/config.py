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
    start_date: str = "2024-01-01"
    end_date: str = "2026-01-31"
    # MOEX ISS поддерживает интервалы 1, 10, 60, 24, 7, 31. Качаем 1-мин и
    # ресэмплируем до bar_minutes - так получаем любой целевой таймфрейм
    # без отдельного API.
    moex_interval: int = 1
    bar_minutes: int = 15
    # MOEX основная сессия: 10:00 - 18:45 МСК = 07:00 - 15:45 UTC.
    session_start_utc: str = "07:00"
    session_end_utc: str = "15:45"
    base_index: str = "IMOEX"
    extra_indexes: tuple[str, ...] = ("RGBI", "RTSI")
    brent_symbol: str = "BZ=F"  # Yahoo Finance тикер Brent
    cbr_currencies: tuple[str, ...] = ("USD", "EUR")
    # Горизонты в барах (15-минутных): 1=15мин, 4=1ч, 16=4ч, 32≈торг.день.
    horizons: tuple[int, ...] = (1, 4, 16, 32)
    window_size: int = 64           # ~2 торговые сессии контекста
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


@dataclass(frozen=True)
class TrainingConfig:
    """Параметры обучения."""

    batch_size: int = 64
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 6
    grad_clip: float = 1.0
    seed: int = 42
    mc_passes: int = 50


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

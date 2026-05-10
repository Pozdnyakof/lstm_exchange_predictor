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
        "VTBR",
        "SBER",
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
    # MOEX currency/selt инструменты с интрадей историей.
    # Эксперимент с PLZL/GMKN: добавление GLDRUB_TOM/SLVRUB_TOM/USD000UTSTOM
    # как фичей edge не дало (USDRUB пустой post-2024-06 из-за санкций;
    # gold/silver intraday — слишком разреженный объём, чтобы давать
    # сигнал). Поэтому по умолчанию отключено. Чтобы включить, передайте
    # codes в DataConfig: metals_fx_codes=("GLDRUB_TOM", "SLVRUB_TOM").
    # Полный список рабочих инструментов задокументирован в data/orchestrator.py.
    metals_fx_codes: tuple[str, ...] = ()
    metals_fx_interval: int = 10
    # === ALGOPACK (платный premium-фид MOEX) ===
    # Микроструктурные данные: aggressive buy/sell разбивка volume,
    # order-flow из заявок (выставленные/снятые), order-book imbalance
    # и spreads. Все 5-мин с 2020-01. Order flow imbalance — один из
    # сильнейших интрадей-предикторов направления в литературе.
    # Включается передачей algopack_products + переменной окружения
    # ALGOPACK_TOKEN. По умолчанию выключено — пайплайн работает и без
    # подписки. Допустимые значения: 'tradestats' | 'orderstats' | 'obstats'.
    algopack_products: tuple[str, ...] = ()
    algopack_market: str = "eq"  # eq | fo | fx
    algopack_request_pause_sec: float = 0.12
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

    # === DLinear / NLinear параметры (architecture="dlinear" | "nlinear") ===
    # DLinear / NLinear (Zeng et al., AAAI 2023) — простые baseline'ы,
    # часто превосходят трансформеры на коротких многомерных рядах.
    linear_seq_len: int = 384  # выровнено с window_size
    linear_kernel_size: int = 25  # для DLinear: окно скользящего среднего

    # === MOMENT (architecture="moment") ===
    # Foundation model AutonLab/MOMENT-1 (ICML 2024). Encoder заморожен,
    # обучается только голова. Требует pip install momentfm.
    # Доступные чекпоинты: AutonLab/MOMENT-1-small (40M params, d=512),
    #                      AutonLab/MOMENT-1-base  (125M, d=768),
    #                      AutonLab/MOMENT-1-large (340M, d=1024).
    moment_checkpoint: str = "AutonLab/MOMENT-1-base"

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

    # === iTransformer (architecture="itransformer") ===
    # Liu et al., ICLR 2024 (arXiv:2310.06625). Inverted attention:
    # каждая variate (фича) — один токен d_model. Self-attention идёт
    # ПО КАНАЛАМ, что явно моделирует межканальные зависимости.
    # Дефолты сильно ужаты после R-0050 collapse-эпизода (см. отчёт):
    # d=64, layers=2 — устраняют overfitting (train/val gap 0.30→<0.05)
    # на 489k samples × 53 features.
    itransformer_seq_len: int = 384
    itransformer_d_model: int = 64
    itransformer_n_layers: int = 2
    itransformer_n_heads: int = 8
    itransformer_d_ff: int = 128
    itransformer_dropout: float = 0.3
    # Stochastic Depth (Huang et al., ECCV 2016): drop_path=0.2 →
    # каждый residual-блок отключается с p=0.2 на тренировке.
    itransformer_drop_path: float = 0.2

    # === Logit Adjustment (Menon et al., ICLR 2021, arXiv:2007.07314) ===
    # Для бинарного BCE с class imbalance: на тренировке logits[h] -=
    # tau · logit(P_train(UP, h)). Смещает оптимум BCE с prior'а в
    # информативную область, доказанно решает predict-the-prior collapse.
    # 0.0 = выключено; 1.0 — Menon's full Bayes-balanced.
    #
    # Sprint 2: понизили 1.0 → 0.5 после R-0051. Tau=1.0 давало overshoot
    # mean prediction +15pp от prior — модель чрезмерно агрессивна на
    # minority при val/test. Menon допускает [0.5, 1.0]; берём середину.
    logit_adjust_tau: float = 0.5


@dataclass(frozen=True)
class TrainingConfig:
    """Параметры обучения.

    Параметры по умолчанию рассчитаны на L4 (40 GB VRAM) / Colab Pro+.
    На T4 (16 GB VRAM) рекомендуется снизить batch_size до 256-512.
    """

    batch_size: int = 2048  # L4 легко выдерживает 2-4K при window=30
    epochs: int = 40
    learning_rate: float = 1e-3
    # Sprint 2: weight_decay 1e-5 → 1e-2 (Kaddour et al. arXiv:2310.04415).
    # При малом effective-N в финансах WD должен быть на 3 порядка больше
    # — 1e-5 фактически не регуляризирует и допускает memorize-overfit
    # (R-0051: train_loss → 0, val_loss → 0.27).
    weight_decay: float = 1e-2
    # Sprint 2: patience 12 → 5. В R-0051 best_epoch разбросан 2-19 и
    # после best модель просто оверфитит на train. 5 эпох запаса
    # достаточно — реальный сигнал ловится в первые 2-5 эпох.
    early_stopping_patience: int = 5
    grad_clip: float = 1.0
    seed: int = 42
    # 100 проходов вместо 50 - точнее оценка эпистемической
    # неопределённости, инференс остаётся быстрым.
    mc_passes: int = 100
    # Stochastic Weight Averaging (Izmailov et al. 2018) - усреднение
    # весов после swa_start_frac × epochs. Даёт более плоский минимум
    # loss landscape и снижает дисперсию по сидам.
    use_swa: bool = True
    # 0.2 вместо 0.5: при early-stopping на эпохе ~10 SWA должен успеть
    # включиться (старт на эпохе 0.2*40=8 при патиенс=12).
    swa_start_frac: float = 0.2
    swa_lr: float = 5e-4
    # T1.2: Huber-δ автоматически вычисляется из распределения таргетов
    # (≈ 2 × median(|y_train|)). Защищает от ошибки когда δ=1.0 слишком
    # большой для нормализованных лог-доходностей масштаба ~1e-3.
    huber_delta_auto: bool = True
    # T2.2: AdamW + CosineAnnealing вместо плоского Adam.
    optimizer: str = "adamw"  # adam | adamw
    scheduler: str = "cosine"  # none | cosine
    # === ImbSAM (Zhou et al., ICCV 2023) ===
    # Sharpness-Aware Minimization применённый только к minority-class.
    # Сокращает train/val gap при class imbalance через регуляризацию на
    # «плоский» минимум именно на minority-сэмплах (Foret ICLR 2021).
    use_imbsam: bool = False
    imbsam_rho: float = 0.05
    # Горизонт-фильтр для определения minority-сэмплов (индекс в horizons).
    # 0 = самый разбалансированный (h=6 при P(UP)=0.29 в R-0050).
    imbsam_horizon_index: int = 0
    # === Mixup (Zhang et al., ICLR 2018) ===
    # β-distribution параметр; 0.0 = выключен. 0.2 — рекомендация
    # Amazon Science 2024 для TS-forecasting.
    mixup_alpha: float = 0.2
    # Вероятность применения mixup к батчу. 0.5 = на каждом 2-м батче
    # — даёт регуляризацию, но оставляет «чистые» примеры.
    mixup_p: float = 0.5
    # === Repulsive Deep Ensembles (D'Angelo NeurIPS 2021) ===
    # Function-space repulsion между членами Deep Ensemble. λ=0 →
    # обычный non-repulsive ensemble. 0.1-0.5 — рекомендуемый диапазон.
    #
    # ``ensemble_repulsion_weight`` используется в:
    #   - DeepEnsembleTrainer (sequential): репульсия от frozen 0..i-1.
    #   - ConcurrentDeepEnsembleTrainer (concurrent): all-pairs SVGD.
    # Симантика немного разная (см. docstring'и), но вес концептуально
    # тот же — просто как множитель RBF-kernel'а в loss'е.
    ensemble_repulsion_weight: float = 0.1
    # === Concurrent ensemble training ===
    # True → все M членов живут на GPU одновременно, DataLoader
    # итерируется 1× за эпоху. Канонический simultaneous SVGD режим
    # (D'Angelo §4.2). VRAM × M, RAM ≈ как у sequential.
    # False (по умолчанию) → sequential обучение, RAM-friendly,
    # но sequential repulsion (frozen-предшественники).
    use_concurrent_ensemble: bool = False


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
    # "bce"       — обычный binary cross-entropy (legacy)
    # "focal"     — focal-loss
    # "asl"       — Asymmetric Loss (Ridnik-Ben-Baruch ICCV 2021), SOTA
    #               на extreme negative dominance; рекомендация после R-0050
    # "composite" — InnerCls + RankIC + Sharpe + Monotone (quant-loss)
    loss_objective: str = "bce"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    # === ASL params (Ridnik & Ben-Baruch, ICCV 2021, arXiv:2009.14119) ===
    asl_gamma_pos: float = 0.0
    asl_gamma_neg: float = 4.0
    asl_clip: float = 0.05
    # === Class-Balanced pos_weight (Cui et al., CVPR 2019) ===
    # Если True — pos_weight вычисляется как (1-β)/(1-β^n_class), β=0.999.
    # Иначе — старый (1-P)/P (агрессивно усиливает minority и провоцирует
    # collapse при P(UP)~0.3 и большом n).
    use_class_balanced_pos_weight: bool = True
    class_balanced_beta: float = 0.999
    # === Composite quant-loss ===
    # Sharpe выключен по умолчанию (после R-0050): scale-invariant Sharpe
    # на коллапсированной модели тождественно 0, не даёт градиента.
    # Включать только когда RankIC/InnerCls уже работают и output-range
    # модели устойчиво > 0.15.
    composite_bce_weight: float = 1.0
    composite_rankic_weight: float = 0.5
    composite_sharpe_weight: float = 0.0
    composite_monotone_weight: float = 0.1
    # Inner classification head для composite: 'bce' | 'focal' | 'asl'.
    # 'asl' по умолчанию — устраняет совпадение minimum'ов всех 4 компонент
    # на константе prior (главная причина collapse в R-0050).
    composite_inner_loss: str = "asl"
    # Uncertainty Weighting (Kendall CVPR 2018): автоматически балансирует
    # разномасштабные компоненты через обучаемый log_var. Заменяет
    # ручной подбор weights.
    composite_uncertainty_weighting: bool = False
    # Случайные портфели:
    n_random_portfolios: int = 1000
    sigma_threshold: float = 3.0

    # === POSITION SIZING ===
    # Режим расчёта объёма позиции при входе:
    # - "equal_split"  — старое поведение: cash / free_slots (≈20% при
    #                    max_positions=5). Полностью аллоцирует капитал.
    # - "fixed_frac"   — фиксированная доля initial_capital на каждый вход
    #                    (см. ``position_size_fraction``). Обычно 5–10%.
    # - "signal_kelly" — sizing пропорционален «edge» сигнала
    #                    (см. ``model.kelly_sizing.signal_kelly_size``);
    #                    fraction ≈ kelly_scale * edge, ограничен
    #                    [0, max_position_size_fraction].
    sizing_mode: str = "equal_split"
    # Доля initial_capital на одну сделку для sizing_mode="fixed_frac".
    # 0.10 = 10% капитала на трейд → можно держать ~10 позиций; снижает
    # дисперсию equity по сравнению с 20% (equal_split при max_positions=5).
    position_size_fraction: float = 0.10
    # Жёсткий cap на размер ОДНОЙ позиции (для всех sizing_mode).
    # Не позволяет Kelly-сайзингу взорваться при сильно перекошенном
    # сигнале: даже если edge огромный, не более ``X * initial_capital``.
    # 1.0 = без cap'а (поведение legacy equal_split, где «бюджет = весь cash
    # делится между свободными слотами»). Ставьте 0.10–0.20 если нужен
    # риск-лимит при fixed_frac/signal_kelly.
    max_position_size_fraction: float = 1.0
    # === EXIT TRIGGERS (intra-bar, на основе high/low внутри бара) ===
    # 0.0 = выключено; >0 = доля entry_price для закрытия раньше horizon.
    # Применяются только если в prices есть колонки ``high``/``low``.
    # Приоритет внутри бара: stop_loss > profit_target > horizon-exit.
    # Stop срабатывает если low <= entry * (1 - stop_loss_pct);
    # target — если high >= entry * (1 + profit_target_pct).
    stop_loss_pct: float = 0.0
    profit_target_pct: float = 0.0
    # === KELLY SIZING ===
    # Параметры для sizing_mode="signal_kelly". edge приблизительно
    # пропорционален (primary - 0.5)·(meta - 0.5); kelly_scale переводит
    # safe «fractional Kelly» (обычно 0.25–0.5 от full Kelly).
    kelly_scale: float = 0.5
    # Базовый «безопасный» порог Primary, ниже которого edge=0
    # (edge = max(primary - kelly_primary_floor, 0)).
    kelly_primary_floor: float = 0.50
    # То же для Meta (если meta-сигнал доступен; иначе игнорируется).
    kelly_meta_floor: float = 0.50


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

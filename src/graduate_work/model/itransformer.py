"""iTransformer (Liu et al., ICLR 2024) — inverted attention для multivariate TS.

[arXiv:2310.06625](https://arxiv.org/abs/2310.06625)

Ключевая идея, отличающая iTransformer от PatchTST/TimeXer:
- В обычных трансформерах для TS attention идёт по **временным** токенам,
  каналы (variates) обрабатываются независимо или через cross-attention.
- iTransformer **инвертирует** это: каждая variate (канал) — один токен
  длины ``d_model``, attention идёт ПО КАНАЛАМ. Это явно моделирует
  межканальные зависимости, что критично для микроструктурных фич
  (OFI, спрэд, imbalance — сильно скоррелированы).

Архитектура:
1. RevIN (опц.) per-instance нормализация.
2. ``VariateEmbedding``: каждая колонка (B, T, 1) → (B, 1, d_model)
   через ``Linear(T → d_model)``.
3. Stack из ``L`` ``EncoderLayer``: self-attention (Q=K=V = variate-токены)
   + FFN.
4. ``Head``: pooling по variate-токенам → ``Linear(d → fc_hidden) →
   GELU → Dropout → Linear(fc_hidden → num_horizons)``.

Совместим с :func:`build_model` и интерфейсом ConvLSTM/TimeXer:
``forward(x: (B, T, F)) -> (B, num_horizons)``.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .mc_dropout import MonteCarloDropout
from .revin import RevIN


class _VariateEmbedding(nn.Module):
    """Проекция (B, T) → (B, d_model) поканально через общий Linear."""

    def __init__(self, seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(int(seq_len), int(d_model))
        self.norm = nn.LayerNorm(int(d_model))
        self.drop = MonteCarloDropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T) -> Linear -> (B, F, d_model).
        h = self.proj(x.transpose(1, 2))
        return self.drop(self.norm(h))


class _DropPath(nn.Module):
    """Stochastic Depth (Huang et al., ECCV 2016): отключает целый
    residual-блок с вероятностью ``p`` на тренировке.

    Сильнее обычного dropout: вместо случайного занулирования отдельных
    активаций — целиком пропускает residual-вклад. Доказанно сокращает
    train/val gap и улучшает обобщение для глубоких трансформеров.
    """

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            msg = f"drop_path p must be in [0, 1), got {p}"
            raise ValueError(msg)
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        # Маска (B, 1, 1, ...) — одна на батч-элемент, общая для всех
        # координат, чтобы блок отключался целиком, не покомпонентно.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        # Деление на keep — re-scaling для unbiased expectation.
        return x * mask / keep


class _EncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer на variate-токенах с DropPath."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.drop_attn = MonteCarloDropout(dropout)
        self.drop_path_attn = _DropPath(drop_path)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            MonteCarloDropout(dropout),
            nn.Linear(d_ff, d_model),
            MonteCarloDropout(dropout),
        )
        self.drop_path_ff = _DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_attn(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.drop_path_attn(self.drop_attn(attn_out))
        h = self.norm_ff(x)
        x = x + self.drop_path_ff(self.ffn(h))
        return x


class ITransformer(nn.Module):
    """iTransformer для классификации/регрессии направления.

    Variate-tokens: каждая фича превращается в один токен размером
    ``cfg.itransformer_d_model``. Self-attention между ВСЕМИ variates
    напрямую моделирует, например, что ``aps_imb_vol_bbo`` сильно
    зависит от ``disb`` и ``spread_bbo``. Это и есть улучшение U1
    относительно TimeXer (где каналы независимы при ``timexer_n_exo=0``).

    Голова: усреднение по variate-токенам → MLP.

    **Logit Adjustment** (Menon et al., ICLR 2021,
    [arXiv:2007.07314](https://arxiv.org/abs/2007.07314)): на тренировке
    из logits вычитается ``tau * log(P(y) / (1 - P(y)))``. Это
    математически смещает оптимум BCE с константного prior'а — без этого
    при class imbalance модель скатывается в predict-the-prior collapse.
    Калибровка делается через :meth:`set_logit_prior`. На инференсе
    эффект отключается (вход в head проходит без adjustment).
    """

    def __init__(
        self,
        input_dim: int,
        num_horizons: int,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        seq = int(cfg.itransformer_seq_len)
        d = int(cfg.itransformer_d_model)
        if seq <= 0:
            msg = f"itransformer_seq_len must be positive, got {seq}"
            raise ValueError(msg)
        if input_dim <= 0:
            msg = f"input_dim must be positive, got {input_dim}"
            raise ValueError(msg)

        self.input_dim = int(input_dim)
        self.seq_len = seq
        self.num_horizons = int(num_horizons)
        self.revin = (
            RevIN(num_features=input_dim, affine=cfg.revin_affine)
            if cfg.use_revin else None
        )

        # DropPath p — стохастическое отключение целого encoder-блока на
        # тренировке (Huang et al., ECCV 2016, "Deep Networks with
        # Stochastic Depth"). Регуляризатор сильнее обычного dropout.
        droppath_p = float(getattr(cfg, "itransformer_drop_path", 0.0))

        self.embed = _VariateEmbedding(
            seq_len=seq,
            d_model=d,
            dropout=float(cfg.itransformer_dropout),
        )
        self.layers = nn.ModuleList([
            _EncoderLayer(
                d_model=d,
                n_heads=int(cfg.itransformer_n_heads),
                d_ff=int(cfg.itransformer_d_ff),
                dropout=float(cfg.itransformer_dropout),
                drop_path=droppath_p,
            )
            for _ in range(int(cfg.itransformer_n_layers))
        ])
        self.norm = nn.LayerNorm(d)

        self.head = nn.Sequential(
            nn.Linear(d, int(cfg.fc_hidden)),
            nn.GELU(),
            MonteCarloDropout(float(cfg.dropout)),
            nn.Linear(int(cfg.fc_hidden), num_horizons),
        )

        # Logit Adjustment: tau=0 = выключено (бэквард-совместимо).
        self.logit_adjust_tau = float(getattr(cfg, "logit_adjust_tau", 0.0))
        # log_prior_logit shape (num_horizons,): logit от P(y=1).
        # set_logit_prior() заполнит его реальными priors из train-данных.
        self.register_buffer(
            "log_prior_logit",
            torch.zeros(self.num_horizons, dtype=torch.float32),
            persistent=False,
        )

    def set_logit_prior(self, p_up: torch.Tensor | list | tuple) -> None:
        """Установить per-horizon prior P(UP) для logit-adjustment.

        Принимает массив длины ``num_horizons`` со значениями в (0, 1).
        Сохраняется как logit(P) в buffer'е ``log_prior_logit``.
        Менон et al. (ICLR 2021): для бинарного BCE из logits вычитается
        ``log(P/(1-P)) = logit(P)``.
        """
        if not isinstance(p_up, torch.Tensor):
            p_up = torch.tensor(p_up, dtype=torch.float32)
        p_up = p_up.float().to(self.log_prior_logit.device)
        if p_up.numel() != self.num_horizons:
            msg = (
                f"prior length {p_up.numel()} != num_horizons "
                f"{self.num_horizons}"
            )
            raise ValueError(msg)
        eps = 1e-6
        p_clamped = p_up.clamp(eps, 1.0 - eps)
        logit = torch.log(p_clamped / (1.0 - p_clamped))
        self.log_prior_logit.copy_(logit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) -> (B, num_horizons).

        Если T < seq_len — left-pad нулями; если T > seq_len — берём
        ПОСЛЕДНИЕ seq_len баров (causal-friendly). На тренировке
        применяется logit-adjustment если ``logit_adjust_tau > 0`` и
        log-prior установлен.
        """
        _, t, f = x.shape
        if f != self.input_dim:
            msg = f"input_dim mismatch: got F={f}, expected {self.input_dim}"
            raise ValueError(msg)
        if self.revin is not None:
            x = self.revin(x)
        if t < self.seq_len:
            pad = self.seq_len - t
            x = nn.functional.pad(x, (0, 0, pad, 0))
        elif t > self.seq_len:
            x = x[:, -self.seq_len:, :]

        tokens = self.embed(x)               # (B, F, d_model)
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)          # (B, d_model)
        logits = self.head(pooled)
        # Logit Adjustment активен ТОЛЬКО на тренировке: смещает оптимум
        # BCE с prior'а, заставляя сеть учиться ранжированию.
        if self.training and self.logit_adjust_tau > 0.0:
            logits = logits - self.logit_adjust_tau * self.log_prior_logit
        return logits

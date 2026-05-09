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


class _EncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer на variate-токенах."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.drop_attn = MonteCarloDropout(dropout)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            MonteCarloDropout(dropout),
            nn.Linear(d_ff, d_model),
            MonteCarloDropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm + residual: устойчивее к глубине, чем post-norm.
        h = self.norm_attn(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.drop_attn(attn_out)
        h = self.norm_ff(x)
        x = x + self.ffn(h)
        return x


class ITransformer(nn.Module):
    """iTransformer для классификации/регрессии направления.

    Variate-tokens: каждая фича превращается в один токен размером
    ``cfg.itransformer_d_model``. Self-attention между ВСЕМИ variates
    напрямую моделирует, например, что ``aps_imb_vol_bbo`` сильно
    зависит от ``disb`` и ``spread_bbo``. Это и есть улучшение U1
    относительно TimeXer (где каналы независимы при ``timexer_n_exo=0``).

    Голова: усреднение по variate-токенам → MLP.
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
        self.revin = (
            RevIN(num_features=input_dim, affine=cfg.revin_affine)
            if cfg.use_revin else None
        )

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) -> (B, num_horizons).

        Если T < seq_len — left-pad нулями; если T > seq_len — берём
        ПОСЛЕДНИЕ seq_len баров (causal-friendly).
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
        # Pool по variate-токенам — простое среднее. iTransformer-paper
        # использует concat+linear, но для нашей задачи (классификация
        # направления, не reconstruct ряда) mean pool работает не хуже
        # и лучше регуляризируется.
        pooled = tokens.mean(dim=1)          # (B, d_model)
        return self.head(pooled)

"""TimeXer (arXiv:2402.19072) — endo/exo Transformer для временных рядов.

Адаптация baseline-модели из исследовательского журнала (R-0023, R09.M).
Patch-based self-attention по эндогенным каналам + опциональная
cross-attention на экзогенные. При ``timexer_n_exo == 0`` вырождается
в PatchTST-подобную чистую self-attention сеть.

Интерфейс совместим с :class:`ConvLstmRegressor`:
    forward: (B, T, F) -> (B, num_horizons)
    classification: возвращает ЛОГИТЫ (sigmoid в инференсе через mc_predict).
    regression:    возвращает прогноз нормализованной лог-доходности.

Использует :class:`MonteCarloDropout` из graduate_work — :func:`set_mc_dropout`
управляет всеми dropout-слоями этой модели.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .mc_dropout import MonteCarloDropout
from .revin import RevIN


class _PatchEmbedding(nn.Module):
    """Развернуть временной ряд на патчи и проецировать в d_model."""

    def __init__(
        self,
        n_vars: int,
        patch_len: int,
        stride: int,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.proj = nn.Linear(self.patch_len * n_vars, d_model)
        self.drop = MonteCarloDropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C). torch.Tensor.unfold по dim=1 -> (B, P, C, patch_len).
        patches = x.unfold(1, self.patch_len, self.stride)
        b, p, c, pl = patches.shape
        patches = patches.reshape(b, p, c * pl)
        return self.drop(self.proj(patches))


class _ExoEmbedding(nn.Module):
    """Сжать каждую экзогенную переменную по времени в один токен d_model."""

    def __init__(self, seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.temporal = nn.Linear(seq_len, d_model)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            MonteCarloDropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_exo) -> (B, n_exo, T) -> (B, n_exo, d_model).
        out = self.temporal(x.transpose(1, 2))
        return self.proj(out)


class _CrossAttention(nn.Module):
    """Query-токены из endo внимают key/value из exo-токенов."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = MonteCarloDropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(q, kv, kv)
        return self.norm(q + self.drop(out))


class _EncoderLayer(nn.Module):
    """Self-attention + (опц.) cross-attention + FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        *,
        use_cross: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            MonteCarloDropout(dropout),
            nn.Linear(d_ff, d_model),
            MonteCarloDropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.cross = (
            _CrossAttention(d_model, n_heads, dropout) if use_cross else None
        )

    def forward(
        self,
        x: torch.Tensor,
        exo: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + out)
        if self.cross is not None and exo is not None:
            x = self.cross(x, exo)
        x = self.norm2(x + self.ffn(x))
        return x


class TimeXer(nn.Module):
    """TimeXer baseline из R-0023 (research project).

    Последние ``cfg.timexer_n_exo`` каналов входа считаются экзогенными,
    остальные — эндогенными. Эндогенные нарезаются на патчи длины
    ``patch_len`` со страйдом ``stride``, через которые работает
    self-attention. Глобальный CLS-токен, attended ко всем патчам и
    (если есть) к экзогенным токенам, идёт в голову.

    Параметры берутся из :class:`ModelConfig`.
    """

    def __init__(
        self,
        input_dim: int,
        num_horizons: int,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        n_exo = int(cfg.timexer_n_exo)
        n_endo = input_dim - n_exo
        if n_endo <= 0:
            msg = (
                f"timexer_n_exo={n_exo} >= input_dim={input_dim};"
                " не осталось эндогенных каналов"
            )
            raise ValueError(msg)
        d = int(cfg.timexer_d_model)
        seq = int(cfg.timexer_seq_len)
        pl = int(cfg.timexer_patch_len)
        st = int(cfg.timexer_stride)
        if (seq - pl) % st != 0 or seq < pl:
            msg = (
                f"timexer_seq_len={seq} не нарезается на патчи "
                f"patch_len={pl}, stride={st}"
            )
            raise ValueError(msg)
        dr = float(cfg.timexer_dropout)
        self.n_endo = n_endo
        self.n_exo = n_exo

        # RevIN — per-instance reversible normalization (R08.B).
        self.revin = (
            RevIN(num_features=input_dim, affine=cfg.revin_affine)
            if cfg.use_revin else None
        )

        n_patches = (seq - pl) // st + 1
        self.endo_embed = _PatchEmbedding(n_endo, pl, st, d, dr)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, d) * 0.02)
        self.global_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        use_cross = n_exo > 0
        self.exo_embed = _ExoEmbedding(seq, d, dr) if use_cross else None

        self.layers = nn.ModuleList([
            _EncoderLayer(
                d,
                n_heads=int(cfg.timexer_n_heads),
                d_ff=int(cfg.timexer_d_ff),
                dropout=dr,
                use_cross=use_cross,
            )
            for _ in range(int(cfg.timexer_n_layers))
        ])
        self.norm = nn.LayerNorm(d)

        # Голова: одна линейка после GELU+Dropout. Для классификации
        # это логиты, для регрессии — нормализованные лог-доходности.
        self.head = nn.Sequential(
            nn.Linear(d, int(cfg.fc_hidden)),
            nn.GELU(),
            MonteCarloDropout(float(cfg.dropout)),
            nn.Linear(int(cfg.fc_hidden), num_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) -> (B, num_horizons)."""
        if self.revin is not None:
            x = self.revin(x)
        b = x.shape[0]
        x_endo = x[:, :, : self.n_endo]
        tokens = self.endo_embed(x_endo)
        glob = self.global_token.expand(b, -1, -1)
        tokens = torch.cat([glob, tokens], dim=1)
        tokens = tokens + self.pos_embed

        exo_tokens: torch.Tensor | None = None
        if self.exo_embed is not None:
            x_exo = x[:, :, self.n_endo:]
            exo_tokens = self.exo_embed(x_exo)

        for layer in self.layers:
            tokens = layer(tokens, exo_tokens)

        # CLS-токен — глобальный summary окна.
        cls = self.norm(tokens[:, 0])
        return self.head(cls)

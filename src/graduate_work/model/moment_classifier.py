"""MOMENT (Goswami et al., ICML 2024) как frozen-encoder + trainable head.

[arXiv:2402.03885](https://arxiv.org/abs/2402.03885)
[HuggingFace AutonLab/MOMENT-1-base/large](https://huggingface.co/AutonLab/MOMENT-1-base)

MOMENT — pretrained foundation model для временных рядов (~125M
параметров для base, ~340M для large, ~40M для small). Обучен на
огромной коллекции univariate-рядов через masked time-series modeling.
Подаём наши N=72 канала как N независимых univariate-рядов длины 384,
получаем embeddings размерности `d_model` (768 для base) на канал,
усредняем по каналам и подаём в маленькую обучаемую голову.

Главное преимущество в нашем сценарии: encoder заморожен, обучается
только голова (~50K параметров). При 108K сэмплах и
~50K тренируемых параметров отношение 2:1 — нормально, без
prediction collapse.

Зависимости:
    pip install momentfm

Если пакет недоступен (например, локальное dev-окружение), модуль
поднимет понятную ошибку при попытке создать класс.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.nn import functional as F

from ..config import ModelConfig
from .mc_dropout import MonteCarloDropout

logger = logging.getLogger(__name__)

# MOMENT требует фиксированный seq_len в зависимости от размера:
# base/large/small — все 512.
_MOMENT_SEQ_LEN = 512


class MomentClassifier(nn.Module):
    """MOMENT encoder (frozen) + multi-horizon classification head.

    Forward:
        x: (B, T, F)
        - переставляет в (B, F, T)  ← MOMENT ждёт channel-first.
        - паддит/обрезает T до 512.
        - для каждого из F каналов получает embedding размера d_model.
        - усредняет по каналам -> (B, d_model).
        - голова: Linear -> GELU -> MCDropout -> Linear(num_horizons).
    """

    def __init__(
        self,
        input_dim: int,
        num_horizons: int,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        try:
            from momentfm import MOMENTPipeline  # noqa: PLC0415
        except ImportError as exc:
            msg = (
                "Чтобы использовать architecture='moment', установите пакет "
                "momentfm: `pip install momentfm`. Веса будут "
                "автоматически скачаны с HuggingFace при первом forward."
            )
            raise ImportError(msg) from exc

        self.input_dim = int(input_dim)
        self.checkpoint = str(cfg.moment_checkpoint)

        logger.info("Loading MOMENT checkpoint: %s", self.checkpoint)
        self.encoder = MOMENTPipeline.from_pretrained(
            self.checkpoint,
            model_kwargs={
                "task_name": "embedding",
                "n_channels": 1,         # обрабатываем по одному каналу за раз
                "freeze_encoder": True,
                "freeze_embedder": True,
                "freeze_head": True,
            },
        )
        self.encoder.init()
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        d_model = int(self.encoder.config.d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, int(cfg.fc_hidden)),
            nn.GELU(),
            MonteCarloDropout(float(cfg.dropout)),
            nn.Linear(int(cfg.fc_hidden), num_horizons),
        )

    @torch.no_grad()
    def _encode_channels(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, F, T) -> (B, F, d_model). Encoder заморожен."""
        bsz, n_ch, seq = x.shape
        # Подгоняем seq_len к 512 (MOMENT требование).
        if seq < _MOMENT_SEQ_LEN:
            pad = _MOMENT_SEQ_LEN - seq
            x = F.pad(x, (pad, 0))     # left-pad нулями
        elif seq > _MOMENT_SEQ_LEN:
            x = x[:, :, -_MOMENT_SEQ_LEN:]
        # MOMENT обрабатывает (B, n_channels=1, T). Развёртываем (B, F)
        # в один батч (B*F, 1, T), считаем embedding, разворачиваем обратно.
        x_flat = x.reshape(bsz * n_ch, 1, _MOMENT_SEQ_LEN)
        out = self.encoder(x_enc=x_flat)
        emb = out.embeddings              # (B*F, d_model)
        return emb.reshape(bsz, n_ch, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        emb = self._encode_channels(x)    # (B, F, d_model)
        # Pool по каналам — простое среднее. Альтернативы (max, attention)
        # можно ввести через cfg.moment_pool, пока не делаем.
        pooled = emb.mean(dim=1)          # (B, d_model)
        return self.head(pooled)

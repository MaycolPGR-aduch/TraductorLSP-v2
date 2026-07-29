"""Arquitecturas del clasificador de señas (PyTorch).

Solo se usa en entrenamiento y export. La aplicación final no importa PyTorch:
consume el modelo exportado a ONNX desde :mod:`senasperu.model.inference`.

Entrada: ``(lote, frames, features)`` con landmarks ya normalizados.
Salida: ``(lote, clases)`` con logits.

El objetivo es <10 MB cuantizado y <50 ms por ventana en CPU, así que ambas
arquitecturas son deliberadamente pequeñas.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from senasperu.config import Config


class PositionalEncoding(nn.Module):
    """Codificación posicional sinusoidal clásica.

    Sin ella el Transformer no distingue el orden de los frames, que es
    justamente lo que separa una seña dinámica de otra con el mismo recorrido
    en sentido inverso.
    """

    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        posicion = torch.arange(max_len).unsqueeze(1).float()
        divisor = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        codificacion = torch.zeros(max_len, dim)
        codificacion[:, 0::2] = torch.sin(posicion * divisor)
        codificacion[:, 1::2] = torch.cos(posicion * divisor)[:, : codificacion[:, 1::2].shape[1]]
        self.register_buffer("encoding", codificacion.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Suma la codificación posicional a la secuencia."""
        return x + self.encoding[:, : x.shape[1], :]


class SignTransformer(nn.Module):
    """Transformer encoder pequeño con agrupación por promedio."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        *,
        layers: int = 3,
        model_dim: int = 128,
        heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        """Args:
        input_size: Largo del vector de features por frame.
        num_classes: Cantidad de señas, incluida la clase de reposo.
        layers: Capas del encoder.
        model_dim: Dimensión interna.
        heads: Cabezas de atención.
        feedforward_dim: Dimensión de la capa feedforward.
        dropout: Dropout aplicado en el encoder y antes del clasificador.
        """
        super().__init__()
        self.input_projection = nn.Linear(input_size, model_dim)
        self.positional = PositionalEncoding(model_dim)
        capa = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(capa, num_layers=layers)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
        x: Tensor ``(lote, frames, features)``.

        Returns:
            Logits ``(lote, clases)``.
        """
        h = self.positional(self.input_projection(x))
        h = self.encoder(h)
        h = self.norm(h.mean(dim=1))
        return self.classifier(self.dropout(h))


class SignBiLSTM(nn.Module):
    """BiLSTM de referencia, para comparar contra el Transformer."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        *,
        layers: int = 2,
        hidden_size: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Igual contrato que :class:`SignTransformer`."""
        salida, _ = self.lstm(x)
        return self.classifier(self.dropout(salida.mean(dim=1)))


def build_model(config: Config, input_size: int, num_classes: int) -> nn.Module:
    """Construye la arquitectura indicada en ``modelo.arquitectura``.

    Raises:
        ValueError: Si la arquitectura configurada no existe.
    """
    arquitectura = str(config.get("modelo.arquitectura", "transformer")).lower()
    if arquitectura == "transformer":
        return SignTransformer(
            input_size,
            num_classes,
            layers=int(config.require("modelo.transformer.capas")),
            model_dim=int(config.require("modelo.transformer.dim_modelo")),
            heads=int(config.require("modelo.transformer.cabezas_atencion")),
            feedforward_dim=int(config.require("modelo.transformer.dim_feedforward")),
            dropout=float(config.require("modelo.transformer.dropout")),
        )
    if arquitectura == "bilstm":
        return SignBiLSTM(
            input_size,
            num_classes,
            layers=int(config.require("modelo.bilstm.capas")),
            hidden_size=int(config.require("modelo.bilstm.dim_oculta")),
            dropout=float(config.require("modelo.bilstm.dropout")),
        )
    raise ValueError(
        f"Arquitectura '{arquitectura}' desconocida. Usa 'transformer' o 'bilstm'."
    )


def count_parameters(model: nn.Module) -> int:
    """Cantidad de parámetros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

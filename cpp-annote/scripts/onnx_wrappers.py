# MIT License
#
# Copyright (c) 2026- pyannote.audio contributors
#
# Small nn.Module wrappers used only for ONNX export (C++ port tooling).

from __future__ import annotations

import torch
import torch.nn as nn

from pyannote.audio.core.model import Model, Specifications
from pyannote.audio.utils.powerset import Powerset


def _as_spec_tuple(specifications: Specifications | tuple[Specifications, ...]) -> tuple[Specifications, ...]:
    if isinstance(specifications, Specifications):
        return (specifications,)
    return specifications


class SegmentationInferenceWrapper(nn.Module):
    """Match `Inference.infer` post-processing: optional Powerset -> hard multilabel.

    This extends the minimal export in https://github.com/pengzhendong/pyannote-onnx so
    that `community-1` (and other powerset segmentation checkpoints) match runtime Python.
    """

    def __init__(self, model: Model, *, skip_conversion: bool = False) -> None:
        super().__init__()
        self.inner = model
        specs = _as_spec_tuple(model.specifications)
        conversions: list[nn.Module] = []
        for s in specs:
            if s.powerset and not skip_conversion:
                conversions.append(Powerset(len(s.classes), s.powerset_max_classes))
            else:
                conversions.append(nn.Identity())
        self.conversions = nn.ModuleList(conversions)
        if len(self.conversions) != 1:
            raise NotImplementedError(
                "ONNX export currently supports a single task (one output branch). "
                f"Got {len(self.conversions)} specification(s)."
            )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        raw = self.inner(waveforms)
        if isinstance(raw, tuple):
            if len(raw) != 1:
                raise NotImplementedError("Multi-output segmentation models are not supported yet.")
            raw = raw[0]
        return self.conversions[0](raw)


class EmbeddingInferenceWrapper(nn.Module):
    """Expose `(waveforms, weights)` for ONNX; matches `Model.__call__(..., weights=...)`."""

    def __init__(self, model: Model) -> None:
        super().__init__()
        self.inner = model

    def forward(self, waveforms: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return self.inner(waveforms, weights=weights)


class EmbeddingFbankOnnxWrapper(nn.Module):
    """WeSpeaker ONNX boundary: ResNet from Kaldi log-fbank (FFT stays outside ONNX).

    Legacy ``torch.onnx.export`` cannot lower ``aten::fft_rfft`` used inside
    ``torchaudio.compliance.kaldi.fbank``. Run the same preprocessing as
    ``BaseWeSpeakerResNet.compute_fbank`` in host code, then feed ``fbank`` here.
    """

    def __init__(self, model: Model) -> None:
        super().__init__()
        if not hasattr(model, "resnet"):
            raise ValueError(
                "Embedding ONNX export expects a WeSpeaker-style model with a ``resnet`` submodule."
            )
        self.resnet = model.resnet

    def forward(self, fbank: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return self.resnet(fbank, weights=weights)[1]

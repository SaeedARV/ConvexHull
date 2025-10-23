"""
Standalone inference pipeline for DeepHullNet-based convex hull prediction.

This module copies the minimal pieces of the original DeepHullNet repository
needed for inference so that the rest of the project no longer depends on the
`DeepHullNet/` directory.  The interface mirrors the existing MVEE and Extents
implementations: instantiate the class with a point cloud and call `compute()`
to obtain the hull vertices predicted by the neural model.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - surfaces at runtime
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None  # type: ignore[assignment]


TOKENS = {"<sos>": 0, "<eos>": 1}
SPECIAL_TOKEN_COUNT = len(TOKENS)
DEEPNET_FEATURES = 5  # (x, y, is_sos, is_eos, is_padding)


def _ensure_torch():
    if torch is None or nn is None or F is None:
        raise ModuleNotFoundError(
            "ConvexHullviaDeepHull requires PyTorch. "
            "Install torch before creating a ConvexHullviaDeepHull instance."
        ) from _TORCH_IMPORT_ERROR


class _AdditivePointer(nn.Module):
    """
    Lightweight pointer mechanism adapted from DeepHullNet.

    The implementation is reformulated to keep batch-first tensors internally
    while preserving the original additive attention scoring.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Preserve original parameter names for checkpoint compatibility.
        self.w1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        decoder_states: torch.Tensor,
        encoder_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            decoder_states: (B, T_dec, H)
            encoder_states: (B, T_enc, H)
            mask: (B, T_dec, T_enc) boolean tensor where False entries are disallowed.
        Returns:
            Log-probabilities over encoder positions for each decoder step.
        """
        # (B, T_dec, T_enc, H)
        encoded = self.w1(encoder_states).unsqueeze(1)
        decoded = self.w2(decoder_states).unsqueeze(2)
        scores = self.v(torch.tanh(encoded + decoded)).squeeze(-1)
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask, neg_inf)
        return F.log_softmax(scores, dim=-1)


class _DeepHullTransformer(nn.Module):
    """
    Transformer encoder/decoder with pointer head for convex hull prediction.
    """

    def __init__(
        self,
        input_dim: int = DEEPNET_FEATURES,
        embed_dim: int = 16,
        hidden_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.pointer = _AdditivePointer(hidden_dim=embed_dim)
        self.num_heads = num_heads

    def forward(
        self, batch_tokens: torch.Tensor, batch_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy decoding used at inference time.

        Args:
            batch_tokens: Padded input tokens, shape (B, T, C).
            batch_lengths: Lengths (including <sos>/<eos>) for each item, shape (B,).
        Returns:
            Tuple of (log pointer scores, argmax indices).
        """
        return self.greedy_decode(batch_tokens, batch_lengths)

    def greedy_decode(
        self, batch_tokens: torch.Tensor, batch_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_seq_len, _ = batch_tokens.shape
        device = batch_tokens.device
        num_steps = max_seq_len - SPECIAL_TOKEN_COUNT

        # Encode sequence (seq_len, batch, embed)
        embedded = self.embedding(batch_tokens)
        encoder_seq = self.encoder(embedded.permute(1, 0, 2))
        encoder_mem = encoder_seq.permute(1, 0, 2)  # (B, T, H)

        # Padding mask for encoder memory (True -> pad)
        positions = torch.arange(max_seq_len, device=device).unsqueeze(0)
        encoder_pad_mask = positions >= batch_lengths.unsqueeze(1)

        # Pointer mask: keeps track of which encoder positions remain selectable.
        pointer_mask = (
            positions.unsqueeze(1).expand(batch_size, num_steps, max_seq_len)
            < batch_lengths.unsqueeze(1).unsqueeze(2)
        )
        pointer_mask[:, :, :SPECIAL_TOKEN_COUNT] = False

        decoder_seq = encoder_seq[:1]  # (1, B, H)
        pointer_scores: List[torch.Tensor] = []
        pointer_indices: List[torch.Tensor] = []

        for step in range(num_steps):
            current_mask = pointer_mask[:, : decoder_seq.shape[0], :]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                decoder_seq.shape[0]
            ).to(device)

            decoder_out = self.decoder(
                decoder_seq,
                encoder_seq,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=encoder_pad_mask,
            )

            decoder_states = decoder_out.permute(1, 0, 2)  # (B, steps, H)
            pointer_log_scores = self.pointer(
                decoder_states, encoder_mem, current_mask
            )
            step_scores = pointer_log_scores[:, -1, :]
            step_indices = torch.argmax(step_scores, dim=-1)

            pointer_scores.append(step_scores)
            pointer_indices.append(step_indices)

            # Prevent selecting the same point twice while keeping special tokens blocked.
            update_mask = torch.zeros(
                batch_size, max_seq_len, dtype=torch.bool, device=device
            )
            update_mask.scatter_(1, step_indices.unsqueeze(1), True)
            update_mask[:, :SPECIAL_TOKEN_COUNT] = False
            pointer_mask[update_mask.unsqueeze(1).expand_as(pointer_mask)] = False

            gathered = encoder_mem.gather(
                1,
                step_indices.view(batch_size, 1, 1).expand(
                    -1, 1, encoder_mem.shape[-1]
                ),
            )
            decoder_seq = torch.cat(
                (decoder_seq, gathered.permute(1, 0, 2)),
                dim=0,
            )

        stacked_scores = torch.stack(pointer_scores, dim=1)
        stacked_indices = torch.stack(pointer_indices, dim=1)
        return stacked_scores, stacked_indices


@dataclass
class DeepHullConfig:
    max_points: int = 50
    checkpoint_path: str = "deephull_transform_best_params.pkl"
    device: Optional[str] = None


class ConvexHullviaDeepHull:
    """
    Wrapper exposing DeepHullNet inference through the same interface as the
    existing convex hull solvers in this project.
    """

    def __init__(self, config: Optional[DeepHullConfig] = None):
        _ensure_torch()
        self.config = config or DeepHullConfig()
        self.max_points = self.config.max_points
        self.device = self._select_device(self.config.device)
        self.model = _DeepHullTransformer()
        self._load_weights(self.config.checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _select_device(requested: Optional[str]) -> torch.device:
        if requested is not None:
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_weights(self, checkpoint_path: str) -> None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"DeepHull checkpoint '{checkpoint_path}' was not found. "
                "Copy the pretrained weights into the project root (see README)."
            )
        payload = torch.load(checkpoint_path, map_location="cpu")
        if "model_state" not in payload:
            raise KeyError(
                f"Expected 'model_state' in checkpoint '{checkpoint_path}'."
            )
        self.model.load_state_dict(payload["model_state"])

    def compute(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Predict convex hull vertices for a set of 2D points.
        """
        hull_indices = self.predict_indices(points)
        pts = np.asarray(points, dtype=np.float32)
        if len(hull_indices) == 0:
            return pts[:0]
        return pts[hull_indices]

    def predict_indices(self, points: Sequence[Sequence[float]]) -> List[int]:
        """
        Returns hull vertex indices in counter-clockwise order according to the model.
        """
        tokens, length = self._format_points(points)
        input_tensor = tokens.unsqueeze(0).to(self.device)
        length_tensor = torch.tensor([length], device=self.device)

        with torch.no_grad():
            _, pointer_indices = self.model(input_tensor, length_tensor)

        raw_indices = pointer_indices[0].cpu().tolist()
        hull_indices: List[int] = []
        seen = set()
        num_points = int(length - SPECIAL_TOKEN_COUNT)

        for idx in raw_indices:
            if idx < SPECIAL_TOKEN_COUNT:
                continue
            point_idx = idx - SPECIAL_TOKEN_COUNT
            if point_idx >= num_points:
                break
            if point_idx in seen:
                break
            seen.add(point_idx)
            hull_indices.append(point_idx)
        return hull_indices

    def _format_points(
        self, points: Sequence[Sequence[float]]
    ) -> Tuple[torch.Tensor, int]:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("ConvexHullviaDeepHull expects an (N, 2) array of points.")
        if pts.shape[0] == 0:
            raise ValueError("At least one point is required.")
        if pts.shape[0] > self.max_points:
            raise ValueError(
                f"DeepHull checkpoint was trained for at most {self.max_points} points, "
                f"but received {pts.shape[0]}."
            )

        tokens = torch.zeros(
            (self.max_points + SPECIAL_TOKEN_COUNT, DEEPNET_FEATURES),
            dtype=torch.float32,
        )
        tokens[TOKENS["<sos>"], 2] = 1.0
        tokens[TOKENS["<eos>"], 3] = 1.0
        tokens[
            SPECIAL_TOKEN_COUNT : SPECIAL_TOKEN_COUNT + pts.shape[0], :2
        ] = torch.from_numpy(pts)
        tokens[
            SPECIAL_TOKEN_COUNT + pts.shape[0] :, 4
        ] = 1.0  # padding flag
        length = pts.shape[0] + SPECIAL_TOKEN_COUNT
        return tokens, length


__all__ = ["ConvexHullviaDeepHull", "DeepHullConfig"]

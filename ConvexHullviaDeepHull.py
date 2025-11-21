from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


torch.manual_seed(41)
np.random.seed(41)


def _to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


class NonNegativeLinear(nn.Module):
    """
    Linear layer with non-negative weights enforced via softplus reparameterisation.
    Optionally constrains each row's norm to maintain a Lipschitz bound.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, max_norm: Optional[float] = None):
        super().__init__()
        self.weight_raw = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.max_norm = max_norm
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight_raw)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _apply_norm_constraint(self, weight: torch.Tensor) -> torch.Tensor:
        if self.max_norm is None:
            return weight
        row_norm = weight.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
        scale = torch.clamp(self.max_norm / row_norm, max=1.0)
        return weight * scale

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = F.softplus(self.weight_raw)
        weight = self._apply_norm_constraint(weight)
        return F.linear(inputs, weight, self.bias)


class LipschitzLinear(nn.Module):
    """
    Linear layer that optionally enforces per-row norm constraints for Lipschitz control.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, max_norm: Optional[float] = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.max_norm = max_norm
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _apply_norm_constraint(self, weight: torch.Tensor) -> torch.Tensor:
        if self.max_norm is None:
            return weight
        row_norm = weight.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
        scale = torch.clamp(self.max_norm / row_norm, max=1.0)
        return weight * scale

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self._apply_norm_constraint(self.weight)
        return F.linear(inputs, weight, self.bias)


class InputConvexNetwork(nn.Module):
    """
    Fully-connected input-convex network as described in DeepHull.
    """

    def __init__(self, input_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer.")

        self.first = nn.Linear(input_dim, hidden_sizes[0])
        self.hidden_from_prev = nn.ModuleList()
        self.hidden_from_input = nn.ModuleList()
        for idx in range(len(hidden_sizes) - 1):
            in_size = hidden_sizes[idx]
            out_size = hidden_sizes[idx + 1]
            self.hidden_from_prev.append(NonNegativeLinear(in_size, out_size))
            self.hidden_from_input.append(NonNegativeLinear(input_dim, out_size))

        self.output_layer = NonNegativeLinear(hidden_sizes[-1], 1, bias=True)
        # Theory-consistent monotone convex activation.
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        z = self.activation(self.first(inputs))
        for layer_prev, layer_input in zip(self.hidden_from_prev, self.hidden_from_input):
            z = self.activation(layer_prev(z) + layer_input(inputs))
        output = self.output_layer(z)
        return output.squeeze(-1)


class OriginalDeepHull(nn.Module):
    """
    Wrapper that keeps the original DeepHull ICNN architecture unchanged.
    """

    def __init__(self, input_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.icnn = InputConvexNetwork(input_dim, hidden_sizes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.icnn(inputs)


class ConvexDeepHull(nn.Module):
    """
    Learning-theoretic convex DeepHull: smooth ICNN with Lipschitz-constrained weights.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        lipschitz_constant: float = 1.0,
        activation: str = "softplus",
    ):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer.")
        self.lipschitz_constant = float(lipschitz_constant)
        if self.lipschitz_constant <= 0:
            raise ValueError("lipschitz_constant must be positive.")
        act = activation.lower()
        if act == "softplus":
            self.activation = nn.Softplus()
        elif act == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError("activation must be 'softplus' or 'relu'.")

        self.first = LipschitzLinear(input_dim, hidden_sizes[0], bias=True, max_norm=self.lipschitz_constant)
        self.hidden_from_prev = nn.ModuleList()
        self.hidden_from_input = nn.ModuleList()
        for idx in range(len(hidden_sizes) - 1):
            in_size = hidden_sizes[idx]
            out_size = hidden_sizes[idx + 1]
            self.hidden_from_prev.append(
                NonNegativeLinear(in_size, out_size, bias=True, max_norm=self.lipschitz_constant)
            )
            self.hidden_from_input.append(
                LipschitzLinear(input_dim, out_size, bias=True, max_norm=self.lipschitz_constant)
            )
        self.output_layer = NonNegativeLinear(hidden_sizes[-1], 1, bias=True, max_norm=self.lipschitz_constant)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        z = self.activation(self.first(inputs))
        for layer_prev, layer_input in zip(self.hidden_from_prev, self.hidden_from_input):
            z = self.activation(layer_prev(z) + layer_input(inputs))
        output = self.output_layer(z)
        return output.squeeze(-1)


def get_deephull_model(method: str, input_dim: int, hidden_sizes: Sequence[int], lipschitz_constant: float) -> nn.Module:
    method_lower = method.lower()
    if method_lower == "original":
        return OriginalDeepHull(input_dim, hidden_sizes)
    if method_lower == "convex":
        return ConvexDeepHull(input_dim, hidden_sizes, lipschitz_constant=lipschitz_constant)
    raise ValueError("method must be 'original' or 'convex'.")


class BoundaryGenerator(nn.Module):
    """
    Generator that proposes challenging negatives close to the decision boundary.
    Outputs live in the normalised data space; radius is set by the trainer based on data scale.
    """

    def __init__(self, noise_dim: int, output_dim: int, hidden_sizes: Sequence[int], radius: float):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("Generator requires hidden layers.")
        layers: List[nn.Module] = []
        last = noise_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last, size))
            layers.append(nn.ReLU())
            last = size
        layers.append(nn.Linear(last, output_dim))
        self.network = nn.Sequential(*layers)
        self.radius = radius

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.network(z)
        return torch.tanh(raw) * self.radius


@dataclass
class TrainingStats:
    epoch: int
    loss_pos: float
    loss_neg: float
    adv_term: float


class ConvexHullviaDeepHull:
    """
    DeepHull solver approximating a convex hull via either the original ICNN or a convex Lipschitz ICNN.
    """

    def __init__(
        self,
        method: str = "original",
        hidden_sizes: Sequence[int] = (64, 64, 64),
        generator_hidden: Sequence[int] = (64, 64),
        noise_dim: int = 16,
        max_epochs: int = 250,
        batch_size: int = 128,
        lambda_neg: float = 2.0,
        generator_steps: int = 1,
        generator_radius: float = 1.2,
        level_set_epsilon: float = 0.05,
        max_grad_norm: Optional[float] = None,
        lipschitz_constant: float = 1.0,
        device: Optional[str] = None,
    ):
        method = method.lower()
        if method not in {"original", "convex"}:
            raise ValueError("method must be 'original' or 'convex'.")
        self.hidden_sizes = tuple(hidden_sizes)
        self.method = method
        self.generator_hidden = tuple(generator_hidden)
        self.noise_dim = int(noise_dim)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.lambda_neg = float(lambda_neg)
        self.generator_steps = int(generator_steps)
        # Fixed theoretical scale: radius = 1.2 * max_i ||x_i|| after normalisation.
        self.generator_radius = 1.2
        self.level_set_epsilon = float(level_set_epsilon)
        self.max_grad_norm = max_grad_norm
        self.lipschitz_constant = float(lipschitz_constant)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model: Optional[nn.Module] = None
        self.generator: Optional[BoundaryGenerator] = None
        self.normaliser_mean: Optional[np.ndarray] = None
        self.normaliser_std: Optional[np.ndarray] = None
        self.training_log: List[TrainingStats] = []
        self._fitted_dim: Optional[int] = None

    def _normalise(self, points: np.ndarray) -> np.ndarray:
        if self.normaliser_mean is None or self.normaliser_std is None:
            raise RuntimeError("Normaliser not initialised.")
        return (points - self.normaliser_mean) / self.normaliser_std

    @staticmethod
    def _compute_normaliser(points: np.ndarray) -> Sequence[np.ndarray]:
        mean = points.mean(axis=0, keepdims=True)
        std = points.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return mean, std

    def _compute_generator_radius(self, normalised: np.ndarray) -> float:
        norms = np.linalg.norm(normalised, axis=1)
        max_norm = float(np.max(norms)) if norms.size else 0.0
        if not np.isfinite(max_norm) or max_norm <= 0:
            max_norm = 1.0
        # Theory-prescribed coverage: radius = 1.2 * max_i ||x_i|| in normalised space.
        return float(self.generator_radius * max_norm)

    def _initialise_models(self, dim: int, generator_radius: float) -> None:
        self.model = get_deephull_model(
            method=self.method,
            input_dim=dim,
            hidden_sizes=self.hidden_sizes,
            lipschitz_constant=self.lipschitz_constant,
        ).to(self.device)
        self.generator = BoundaryGenerator(
            noise_dim=self.noise_dim,
            output_dim=dim,
            hidden_sizes=self.generator_hidden,
            radius=generator_radius,
        ).to(self.device)

    def fit(self, points: Sequence[Sequence[float]]) -> None:
        array = np.asarray(points, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("Input points must be a 2-D array.")
        if array.shape[0] < array.shape[1] + 1:
            raise ValueError("Need at least d+1 points to approximate a convex hull.")

        n_points, dim = array.shape
        self.normaliser_mean, self.normaliser_std = self._compute_normaliser(array)
        normalised = self._normalise(array)
        # Scale generator radius to the data: Option (A) from the spec.
        generator_radius = self._compute_generator_radius(normalised)

        self._initialise_models(dim, generator_radius=generator_radius)
        assert self.model is not None and self.generator is not None
        hull_net = self.model
        generator = self.generator

        dataset = torch.utils.data.TensorDataset(_to_tensor(normalised, self.device))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.batch_size, n_points),
            shuffle=True,
            drop_last=False,
        )

        hull_opt = torch.optim.Adam(hull_net.parameters(), lr=1e-3)
        generator_opt = torch.optim.Adam(generator.parameters(), lr=1e-3)

        hull_net.train()
        generator.train()
        self.training_log = []

        for epoch in range(self.max_epochs):
            pos_losses: List[float] = []
            neg_losses: List[float] = []
            adv_terms: List[float] = []

            for (batch_pos,) in loader:
                batch_pos = batch_pos.to(self.device)
                batch_size = batch_pos.shape[0]

                # Generator ascends hull logits to locate hard negatives close to the boundary.
                for _ in range(self.generator_steps):
                    noise = torch.randn(batch_size, self.noise_dim, device=self.device)
                    generator_opt.zero_grad(set_to_none=True)
                    hull_opt.zero_grad(set_to_none=True)
                    generated = generator(noise)
                    logits = hull_net(generated)
                    generator_loss = -torch.mean(logits)
                    generator_loss.backward()
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(generator.parameters(), self.max_grad_norm)
                    generator_opt.step()
                    hull_opt.zero_grad(set_to_none=True)

                with torch.no_grad():
                    noise = torch.randn(batch_size, self.noise_dim, device=self.device)
                    hard_negatives = generator(noise)

                negatives = hard_negatives

                hull_opt.zero_grad(set_to_none=True)
                logits_pos = hull_net(batch_pos)
                logits_neg = hull_net(negatives)
                labels_pos = torch.zeros_like(logits_pos)
                labels_neg = torch.ones_like(logits_neg)

                loss_pos = F.binary_cross_entropy_with_logits(logits_pos, labels_pos)
                loss_neg = F.binary_cross_entropy_with_logits(logits_neg, labels_neg)
                # Theory-specified DeepHull objective: L_pos + lambda * L_neg (no extra boundary term).
                loss = loss_pos + self.lambda_neg * loss_neg

                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(hull_net.parameters(), self.max_grad_norm)
                hull_opt.step()

                pos_losses.append(float(loss_pos.item()))
                neg_losses.append(float(loss_neg.item()))
                adv_terms.append(float(torch.sigmoid(logits_neg.detach()).mean().item()))

            if not pos_losses:
                continue
            self.training_log.append(
                TrainingStats(
                    epoch=epoch,
                    loss_pos=float(np.mean(pos_losses)),
                    loss_neg=float(np.mean(neg_losses)),
                    adv_term=float(np.mean(adv_terms)),
                )
            )

        hull_net.eval()
        generator.eval()
        self._fitted_dim = dim

    def _predict_scores(self, points: np.ndarray) -> np.ndarray:
        if self.model is None or self.normaliser_mean is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        normalised = self._normalise(points)
        tensor = _to_tensor(normalised, self.device)
        with torch.no_grad():
            scores = self.model(tensor).cpu().numpy()
        return scores.astype(np.float64, copy=False)

    def predict_indices(self, points: Sequence[Sequence[float]]) -> List[int]:
        points_array = np.asarray(points, dtype=np.float32)
        if points_array.ndim != 2:
            raise ValueError("Input points must be two-dimensional array-like.")

        dim = points_array.shape[1]
        self.fit(points_array)

        scores = self._predict_scores(points_array)
        threshold = -self.level_set_epsilon
        candidate_indices = [int(i) for i, score in enumerate(scores) if score >= threshold]
        if len(candidate_indices) < dim + 1:
            sorted_indices = np.argsort(scores)[::-1]
            top_k = min(points_array.shape[0], max(dim + 1, len(candidate_indices)))
            candidate_indices = [int(idx) for idx in sorted_indices[:top_k]]

        return candidate_indices


if __name__ == "__main__":
    from common import sample_input

    data = sample_input()
    solver = ConvexHullviaDeepHull()
    indices = solver.predict_indices(data)
    print("Estimated hull indices:", indices)

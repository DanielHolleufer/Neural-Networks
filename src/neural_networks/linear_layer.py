import numpy as np
import numpy.typing as npt
from typing import Protocol


class WeightInitializer(Protocol):
    def __call__(self, input_dim: int, output_dim: int, dtype) -> npt.NDArray: ...


class Linear:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        weight_initializer: WeightInitializer,
        dtype: npt.DTypeLike = np.float64,
    ) -> None:
        self.weights = weight_initializer(input_dim, output_dim, dtype)
        self.biases = np.zeros(output_dim, dtype=dtype)
        self.input = None
        self.weight_derivatives = None
        self.bias_derivatives = None

    def forward(self, input: npt.NDArray) -> npt.NDArray:
        return input @ self.weights + self.biases


class FromMatrix:
    def __init__(self, weights: npt.NDArray) -> None:
        self.weights = weights

    def __call__(self, input_dim: int, output_dim: int, dtype: npt.DTypeLike) -> npt.NDArray:
        if not self.weights.shape == (input_dim, output_dim):
            raise ValueError(
                f"FromMatrix: "
                f"Weights matrix shape does not match layer input_dim and output_dim. "
                f"Weights matrix shape: {self.weights.shape} "
                f"Layer (input_dim, output_dim): ({input_dim}, {output_dim})."
            )

        return self.weights.astype(dtype, copy=True)


class UniformGlorot:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, input_dim: int, output_dim: int, dtype: npt.DTypeLike) -> npt.NDArray:
        limit = np.sqrt(6 / (input_dim + output_dim))
        return self.rng.uniform(-limit, limit, (input_dim, output_dim)).astype(dtype)


class UniformHe:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, input_dim: int, output_dim: int, dtype: npt.DTypeLike) -> npt.NDArray:
        limit = np.sqrt(6 / input_dim)
        return self.rng.uniform(-limit, limit, (input_dim, output_dim)).astype(dtype)

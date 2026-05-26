import numpy as np
import numpy.typing as npt
from typing import cast, Protocol


class WeightInitializer[F: np.floating = np.float64](Protocol):
    def __call__(
        self,
        input_dim: int,
        output_dim: int,
        seed: int | None = None,
        dtype: type[F] = np.float64,
    ) -> npt.NDArray[F]: ...


class Linear[F: np.floating = np.float64]:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        weight_initializer: WeightInitializer[F],
        seed: int | None = None,
        dtype: type[F] = np.float64,
    ) -> None:
        self.weights: npt.NDArray[F] = weight_initializer(
            input_dim, output_dim, seed=seed, dtype=dtype
        )
        self.biases: npt.NDArray[F] = np.zeros(output_dim, dtype=dtype)
        self.input: npt.NDArray[F] | None = None
        self.weight_derivatives: npt.NDArray[F] | None = None
        self.bias_derivatives: npt.NDArray[F] | None = None

    def forward(self, input: npt.NDArray[F]) -> npt.NDArray[F]:
        return cast(npt.NDArray[F], input @ self.weights + self.biases)


def uniform_glorot[F: np.floating = np.float64](
    input_dim: int,
    output_dim: int,
    seed: int | None = None,
    dtype: type[F] = np.float64,
) -> npt.NDArray[F]:
    rng = np.random.default_rng(seed)
    limit = np.sqrt(6 / (input_dim + output_dim))
    return rng.uniform(-limit, limit, (input_dim, output_dim)).astype(dtype)


def uniform_he[F: np.floating = np.float64](
    input_dim: int,
    output_dim: int,
    seed: int | None = None,
    dtype: type[F] = np.float64,
) -> npt.NDArray[F]:
    rng = np.random.default_rng(seed)
    limit = np.sqrt(6 / input_dim)
    return rng.uniform(-limit, limit, (input_dim, output_dim)).astype(dtype)

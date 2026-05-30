import pytest
import numpy as np
from neural_networks.linear_layer import Linear, FromMatrix


N = 4

VEC_ASCENDING = np.arange(N)
VEC_DESCENDING = np.flip(np.arange(N))
VEC_ZEROS = np.zeros(N)
VEC_ONES = np.ones(N)
VEC_SUMS = np.full(N, np.sum(np.arange(N)))

BATCH = np.array([VEC_ASCENDING, VEC_DESCENDING, VEC_ZEROS, VEC_ONES])
BATCH_FLIPPED = np.array([VEC_DESCENDING, VEC_ASCENDING, VEC_ZEROS, VEC_ONES])
BATCH_ZEROS = np.array([VEC_ZEROS, VEC_ZEROS, VEC_ZEROS, VEC_ZEROS])
BATCH_SUMS = np.array([VEC_SUMS, VEC_SUMS, VEC_ZEROS, N * VEC_ONES])

MAT_DIAGONAL = np.eye(N)
MAT_ANTI_DIAGONAL = np.fliplr(np.eye(N))
MAT_ZEROS = np.zeros((N, N))
MAT_ONES = np.ones((N, N))


class TestLinearForward:
    @pytest.mark.parametrize(
        "weights, expected",
        [
            (MAT_DIAGONAL, BATCH),
            (MAT_ANTI_DIAGONAL, BATCH_FLIPPED),
            (MAT_ZEROS, BATCH_ZEROS),
            (MAT_ONES, BATCH_SUMS),
        ],
    )
    def test_forward_without_bias(self, weights, expected):
        layer = Linear(N, N, FromMatrix(weights))
        y = layer.forward(BATCH)
        np.testing.assert_array_equal(y, expected)

    @pytest.mark.parametrize("biases", [VEC_ONES, VEC_ASCENDING])
    def test_forward_with_bias(self, biases):
        layer = Linear(N, N, FromMatrix(MAT_DIAGONAL))
        layer.biases = biases
        expected = np.array(
            [VEC_ASCENDING + biases, VEC_DESCENDING + biases, VEC_ZEROS + biases, VEC_ONES + biases]
        )
        y = layer.forward(BATCH)
        np.testing.assert_array_equal(y, expected)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_forward_dtype(self, dtype):
        layer = Linear(N, N, FromMatrix(MAT_DIAGONAL), dtype=dtype)
        y = layer.forward(VEC_ASCENDING.astype(dtype))
        assert type(y[1]) is dtype

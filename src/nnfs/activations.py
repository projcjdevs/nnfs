from typing import Optional
import numpy as np
from nnfs.module import Module

class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()
        self._cache_input: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            raise TypeError("Input x must be a NumPy ndarray.")
        self._cache_input = x
        return np.maximum(0.0, x)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._cache_input is None:
            raise RuntimeError(
                "Cannot call backward before forward. "
                "ReLU needs cached input from the forward pass. "
            )

        if not isinstance(grad_out, np.ndarray):
            raise ValueError(
                f"Expected grad_out shape {self._cache_input.shape}, "
                f"but got shape {grad_out.shape}."
            )

        mask = (self._cache_input > 0).astype(grad_out.dtype)

        return grad_out * mask
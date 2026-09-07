from typing import Optional
import numpy as np
from nnfs.module import Module

class Linear(Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__()

        if in_features <= 0:
            raise ValueError('in_features must be a positive integer.')
        if out_features <= 0:
            raise ValueError('out_features must be a positive integer.')    

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self._cache_input: Optional[np.ndarray] = None

        generator = rng if rng is not None else np.random.default_rng()

        limit = np.sqrt(6.0 / (in_features + out_features))
        
        self._parameters["weight"] = generator.uniform(
            low=-limit,
            high=limit,
            size=(in_features, out_features),
        ).astype(np.float64)

        self._grads['weight'] = np.zeros_like(self._parameters['weight'])

        if self.use_bias:
            self._parameters["bias"] = np.zeros(out_features, dtype=np.float64)
            self._grads['bias'] = np.zeros_like(self._parameters['bias'])

    @property
    def weight(self) -> np.ndarray:
        return self._parameters["weight"]

    @property 
    def bias(self) -> Optional[np.ndarray]:
        if not self.use_bias:
            return None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            raise TypeError("Input x must be a NumPy ndarray.")

        if x.ndim != 2:
            raise ValueError(
                f"Expected input x to be 2D with shape "
                f"({x.shape[0]}, {self.in_features}), but got shape {x.shape}."
            )

        if x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected input feature dimension {self.in_features}, "
                f"but got {x.shape[1]}."
            )

        self._cache_input = x

        output = x @ self.weight

        if self.use_bias:
            output = output + self._parameters["bias"]

        return output

    def backward(self, grad_out: np,ndarray) -> np.ndarray:
        if self._cache_input is None:
            raise RuntimeError(
                "Cannot call backward before forward. "
                "The layer needs cached input from the forward pass. "
            )

        if not isinstance(grad_out, np.ndarray):
            raise TypeError("grad_out must be a NumPy ndarray. ")

        if grad_out.ndim != 2:
            raise ValueError(
                f"Expected grad_out to be 2D with shape "
                f"(batch_size, {self.out_features}), but got shape {grad_out.shape}."
            )

        x = self._cache_input

        if grad_out.ndim != 2:
            raise ValueError(
                f"Expected grad_out to be 2D with shape "
                f"(batch_size, {self.out_features}), but got shape {grad_out.shape}."
            )

        x = self._cache_input

        if grad_out.shape[0] != x.shape[0]:
            raise ValueError(
                f"Expected grad_out to be 2D with shape "
                f"but grad_out batch size is {grad_out.shape[0]}. "
            )

        if grad_out.shape[1] != self.out_features:
            raise ValueError(
                f"Expectd grad_out features dimension {self.out_features}, "
                f"but got {grad_out.shape[1]}."
            )

        self._grads["weight"] = x.T @ grad_out
        if self.use_bias:
            self._grads["bias"] = np.sum(grad_out, axis=0)

        grad_input = grad_out @ self.weight.T

        return grad_input
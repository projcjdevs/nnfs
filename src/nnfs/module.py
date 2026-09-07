import numpy as np
from abc import ABC, abstractmethod
from typing import List, Any, Dict

class Module(ABC):

    def __init__(self):
        self._parameters: Dict[str, np.ndarray] = {}
        self._grads: Dict[str, np.ndarray] = {}

        self.training: bool = True

    @abstractmethod
    def forward(self, x: Any) -> Any:
        pass

    @abstractmethod
    def backward(self, grad_out: Any) -> Any:
        pass

    def __call__(self, x: Any) -> Any:
        return self.forward(x)

    def parameters(self) -> List[np.ndarray]:
        return list(self._parameters.values())

    def grads(self) -> List[np.ndarray]:
        return list(self._grads.values())

    def zero_grad(self) -> None:
        for key in self._grds:
            if self._grads[key] is not None:
                self._grads[key] = np.zeros_like(self._grads[key])
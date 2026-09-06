# nnfs: Neural Network From Scratch

A lightweight, object-oriented deep learning framework built entirely from scratch using raw NumPy.

## Project Philosophy

Modern deep learning frameworks like PyTorch and TensorFlow abstract away the mechanics of forward propagation and backpropagation via automatic differentiation. While highly efficient for production, this abstraction hides the underlying linear algebra and calculus.

`nnfs` is engineered to strip away that abstraction. It is built to deeply understand how gradients flow through matrix multiplications, how Jacobians are constructed, and how optimization algorithms update weights.

Rather than functioning as a step-by-step educational notebook, `nnfs` is structured as a production-grade Python package. It enforces strict Object-Oriented design patterns, modularity, and interface contracts, mirroring the architectural standards of industry-leading ML frameworks.

## Architectural Design

The framework rejects complex dynamic computational graph tapes in favor of a deterministic, layer-by-layer sequential architecture. This design ensures mathematical clarity and straightforward debugging.

### The Module Contract

Every component in `nnfs` (Layers, Activations, Loss Functions) inherits from an Abstract Base Class (`Module`). This enforces a strict contract:

- `forward(x)`: Defines the mathematical transformation of the input.
- `backward(grad_out)`: Defines the calculus derivative and gradient propagation.
- `parameters()`: Exposes learnable weights to optimizers.
- `grads()`: Exposes stored gradients corresponding to learnable parameters.
- `zero_grad()`: Manages state clearing to prevent gradient accumulation across batches.

## Mathematical Verification

Because analytical calculus (backpropagation) is prone to implementation errors, `nnfs` employs rigorous numerical verification.

The testing suite utilizes Gradient Checking via Finite Differences. By perturbing weights by a microscopic epsilon and measuring the exact change in the loss function, the framework compares the numerical gradient against the analytical gradient computed by the `backward()` methods. This ensures the mathematical engine is correct without relying on black-box autograd systems.

## Repository Structure

```text
nnfs/
├── src/
│   └── nnfs/
│       ├── __init__.py
│       └── module.py
├── tests/
├── pyproject.toml
├── .gitignore
└── README.md
```

## Installation & Setup

This project uses modern Python packaging standards (`pyproject.toml`). It is highly recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/yourusername/nnfs.git
cd nnfs

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in editable mode with development tools
pip install -e ".[dev]"
```

To verify that the editable installation works:

```bash
python -c "import nnfs; print(nnfs.__file__)"
```

## Development Roadmap

The framework is being built incrementally through strictly defined engineering sprints:

- [x] Sprint 1: Core Engine & Abstract Interfaces (`Module` contract)
- [ ] Sprint 2: Linear Algebra Engine (`Linear` layer & Gradient Checking)
- [ ] Sprint 3: Non-Linearity Mechanics (`ReLU`, `Softmax` derivatives)
- [ ] Sprint 4: Error Quantification (`MSE`, `CrossEntropy` loss functions)
- [ ] Sprint 5: Optimization Strategies (`SGD` implementation)
- [ ] Sprint 6: Integration & Application (Training an MLP on MNIST)

## Testing Strategy

Testing in `nnfs` is centered around mathematical correctness.

The primary testing method is numerical gradient checking:

1. Compute the analytical gradient using the layer's `backward()` method.
2. Approximate the gradient numerically using finite differences:
   - Increase a weight by a small epsilon.
   - Decrease the same weight by the same epsilon.
   - Measure the change in loss.
3. Compare both gradients using a tolerance threshold.

If the analytical gradient closely matches the numerical gradient, the backpropagation implementation is considered mathematically valid.

## Design Principles

- Explicit over implicit.
- Modular components over monolithic scripts.
- Mathematical correctness over premature optimization.
- Engineering discipline over tutorial-style copying.
- Deep understanding over framework convenience.

## Tech Stack

- Language: Python 3.9+
- Math Engine: NumPy
- Testing: PyTest
- Packaging: PEP 621 (`pyproject.toml`)
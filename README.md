# Inverse Physics-Informed Neural Networks for Transport Models in Porous Materials

This repository contains code implementing Physics-Informed Neural Networks (PINNs) for solving inverse problems in transport phenomena, specifically focusing on advection-diffusion-reaction equations in porous media. The implementation is based on the methodology described in Berardi et al. (Inverse Physics-Informed Neural Networks for transport models in porous materials) and has been extended to recover spatially varying parameters (velocity and diffusion coefficient profiles).

## Overview

Physics-Informed Neural Networks (PINNs) offer a powerful framework for solving forward and inverse problems involving partial differential equations (PDEs). This code implements two approaches:

1. **Original Implementation**: PINN for solving inverse problems with constant parameters: diffusion coefficient, advection velocity, reaction constant) (```ad_PINN.ipynb```).
2. **Extended Implementation**: Recovery of spatially varying parameters ($u(x)$) and ($D(x)$) using separate neural networks (```ad_inverse_PINN_varying.py```).

## Mathematical Formulation

### Governing Equation

The code solves the advection-diffusion-reaction equation:

$$
\beta_0 \frac{\partial c}{\partial t} + \frac{\partial}{\partial x}(u c) - \frac{\partial}{\partial x}\left(D \frac{\partial c}{\partial x}\right) = R(c)
$$

where:
- \(c(x,t)\): concentration
- \(\beta_0\): porosity (constant)
- \(u(x)\): advection velocity
- \(D(x)\): diffusion coefficient
- \(R(c)\): reaction term

### Boundary Conditions

- No-flux boundary conditions at both ends:
  $$
  \left. \left(u c - D \frac{\partial c}{\partial x}\right)\right|_{x=0} = 0, \quad \left. \left(u c - D \frac{\partial c}{\partial x}\right)\right|_{x=1} = 0
  $$

### Reaction Models

Several reaction models are implemented:
- **Michaelis-Menten**: \(R(c) = \sigma \frac{c^{n_f}}{\sigma_2 + c^{n_b}}\)
- **Linear**: \(R(c) = \sigma c\)
- **Quadratic**: \(R(c) = \sigma c^2\)
- **Polynomial**: \(R(c) = \sigma c^{n_f}(1-c)^{n_b}\)

## Repository Structure

```
├── data/                        # Data directory for all test cases
├── output/                      # Results from simulations
├── ad_PINN.ipynb                # Jupyter notebook: test pure diffusion model (original)
├── ad_PINN_varying.ipynb        # Jupyter notebook: inverse PINN with u(x) and D(x)
├── ad_inverse_PINN_varying.py   # Organized Python script for spatially varying parameters
├── ad_varying_data.ipynb        # Jupyter notebook: synthetic data generated with Crank-Nicolson method
├── License                      # MIT License
└── README.md
```

## Original Implementation: Constant Parameters

The original code (`original_pinn.py`) recovers constant parameters (diffusion coefficient \(D\), advection velocity \(u\), and reaction constant \(\sigma\)) from concentration data.

### Key Features

- **Single neural network** for concentration field \(c(x,t)\)
- **Trainable parameters**: \(D\), \(u\), \(\sigma\) (can be fixed or trained)
- **Adaptive loss weighting** for parameter training
- **Multiple learning rate schedules** (piecewise constant, exponential, polynomial)

### Loss Function Components

The total loss is a weighted sum of:
- **PDE residual**: \(\| \beta_0 c_t + (u c)_x - (D c_x)_x - R(c) \|^2\)
- **Data fitting**: \(\| c_{\text{model}} - c_{\text{data}} \|^2\)
- **Initial condition**: \(\| c(x,0) - c_0(x) \|^2\)
- **Boundary condition**: \(\| \text{flux}(0) \|^2 + \| \text{flux}(1) \|^2\)

## Extended Implementation: Spatially Varying Parameters

The extended code (`spatially_varying_pinn.py`) recovers spatially varying profiles \(u(x)\) and \(D(x)\) using separate neural networks.

### Key Innovations

- **Three neural networks**:
  1. **Concentration network**: \(c(x,t)\)
  2. **Velocity network**: \(u(x)\)
  3. **Diffusion network**: \(D(x)\)

- **Two-phase training strategy**:
  1. **Phase 1** (2000 epochs): Train only the concentration network to fit data (freeze u/D networks)
  2. **Phase 2** (5000 epochs): Joint training with all networks

- **Smoothness regularization**: \(\| \nabla u \|^2 + \| \nabla D \|^2\) to prevent overfitting
- **Anchor loss**: Regularizes the mean values of \(u(x)\) and \(D(x)\) to known means

### Loss Function (Extended)

Additional loss components:
- **Smoothness regularization**: \(\| \partial u/\partial x \|^2 + \| \partial D/\partial x \|^2\)
- **Anchor loss**: \(\| \langle u \rangle - \bar{u}_{\text{true}} \|^2 + \| \langle D \rangle - \bar{D}_{\text{true}} \|^2\)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{berardi2023inverse,
  title={Inverse Physics-Informed Neural Networks for transport models in porous materials},
  author={Berardi, M. and others},
  journal={Journal of Computational Physics},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the work of Berardi et al. on inverse PINNs
- Extended implementation inspired by recent advances in physics-informed machine learning for parameter field estimation

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.
2. Berardi, M., et al. (2023). Inverse Physics-Informed Neural Networks for transport models in porous materials.
3. Wang, S., et al. (2021). When and why PINNs fail to train: A neural tangent kernel perspective. Journal of Computational Physics, 449, 110768.
```

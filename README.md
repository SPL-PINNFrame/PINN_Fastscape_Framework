# PINN Fastscape Framework

## Overview

This project implements a Physics-Informed Neural Network (PINN) framework designed to act as a differentiable surrogate model for the Fastscape landscape evolution model. The primary goals are:

1.  **Accelerated Simulation**: Leverage the inference speed of neural networks for faster landscape evolution predictions compared to the original Fastscape simulator.
2.  **Differentiability**: Enable gradient-based optimization for inverse problems, such as inferring uplift rate fields from observed topography, by making the evolution process differentiable via the PINN.
3.  **Physics Integration**: Incorporate the governing partial differential equations (PDEs) of landscape evolution (stream power erosion, hillslope diffusion) directly into the PINN's loss function to improve physical realism and generalization.

The framework uses Fastscape (via xarray-simlab) solely for generating training and validation data. The PINN itself learns to approximate the solution to the governing PDE.

## Project Structure

```
PINN_Fastscape_Framework/
├── configs/               # Configuration files (YAML)
│   ├── data_gen_config.yaml # Parameters for data generation
│   ├── train_config.yaml    # Parameters for model training
│   └── optimize_config.yaml # Parameters for optimization/inverse problems
├── data/                  # Data storage (not tracked by git by default)
│   ├── processed/         # Processed data ready for DataLoader (e.g., train/val/test splits)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── raw/               # Optional: Raw output from Fastscape simulations
├── external/              # External libraries (manual placement required)
│   ├── fastscape/         # Fastscape Python code (using xarray-simlab)
│   └── fastscapelib-fortran/ # Fastscape Fortran core library
├── logs/                  # Log files generated during runs (not tracked by git)
├── results/               # Output directory for runs (models, plots, etc.)
│   └── [run_name]/        # Subdirectory for each specific run
│       ├── checkpoints/   # Saved model checkpoints
│       ├── logs/          # Run-specific logs
│       └── optimize_output/ # Output from optimization runs
├── scripts/               # Executable Python scripts
│   ├── generate_data.py   # Script to generate simulation data using Fastscape
│   ├── train.py           # Script to train the PINN model
│   └── optimize.py        # Script to run optimization/inverse problems using a trained PINN
├── src/                   # Source code for the PINN framework
│   ├── __init__.py
│   ├── data_utils.py      # Dataset and DataLoader definitions
│   ├── losses.py          # Loss function implementations (Data, PDE Residual)
│   ├── models.py          # PINN model architectures (e.g., MLP_PINN)
│   ├── optimizer_utils.py # Utilities for the optimization script
│   ├── physics.py         # Differentiable implementations of physics components (derivatives, PDE terms)
│   └── utils.py           # Utility functions (logging, config loading, seeding, etc.)
├── requirements.txt       # Python package dependencies
└── README.md              # This file
```

## Setup

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd PINN_Fastscape_Framework
    ```

2.  **Place External Libraries**:
    *   Manually copy your existing `fastscape` and `fastscapelib-fortran` directories into the `PINN_Fastscape_Framework/external/` directory.

3.  **Create Environment**: It is highly recommended to use a Conda environment.
    ```bash
    conda create -n pinn_fastscape python=3.9 -y # Or desired Python version
    conda activate pinn_fastscape
    ```

4.  **Install Dependencies**:
    *   **Compile Fortran Core**: Navigate to the Fortran library directory and compile it. This typically requires a Fortran compiler (like `gfortran`) and `f2py` (usually installed with `numpy`).
        ```bash
        cd external/fastscapelib-fortran
        pip install --no-build-isolation --editable . # Installs in editable mode, compiling Fortran code
        cd ../.. # Return to project root
        ```
        *Note: Compilation steps might vary based on your system and the library's setup.*
    *   **Install Python Packages**: Install packages listed in `requirements.txt`.
        ```bash
        pip install -r requirements.txt
        ```
        Ensure you have the correct PyTorch version installed for your hardware (CPU/GPU). Visit the [PyTorch website](https://pytorch.org/) for specific installation commands.

## Workflow

1.  **Configure Data Generation**: Edit `configs/data_gen_config.yaml` to define the desired simulation parameters, ranges for sampling, number of simulations, and output directories. Pay attention to mapping parameters to `xarray-simlab` variable names (e.g., `kf` -> `spl__k_coef`).

2.  **Generate Data**: Run the data generation script. This will use `xarray-simlab` to call Fastscape and save the results as `.pt` files in the specified processed data directory (split into train/val/test).
    ```bash
    python scripts/generate_data.py --config configs/data_gen_config.yaml
    ```
    *Note: This step requires a working Fastscape installation (including the compiled Fortran core).*

3.  **Configure Training**: Edit `configs/train_config.yaml`. Define the run name, model architecture, optimizer, learning rate, loss weights, physics parameters (including domain size and the `drainage_area_method`), data paths, etc. **Crucially, update the `drainage_area_method` parameter once a differentiable method is implemented in `src/physics.py`.**

4.  **Train Model**: Run the training script.
    ```bash
    python scripts/train.py --config configs/train_config.yaml
    ```
    Logs and model checkpoints will be saved under `results/[run_name]/`.

5.  **Configure Optimization**: Edit `configs/optimize_config.yaml`. Specify the path to the trained model checkpoint, the target observation data, the parameter to optimize (e.g., `uplift_rate`), optimization settings, and fixed physics parameters.

6.  **Run Optimization**: Execute the optimization script.
    ```bash
    python scripts/optimize.py --config configs/optimize_config.yaml
    ```
    The optimized parameter(s) will be saved in the run's output directory.

## Critical Implementation Notes & TODOs

*   **Differentiable Drainage Area**: The function `calculate_drainage_area_differentiable` in `src/physics.py` is currently a placeholder. A robust, differentiable method for calculating drainage area (e.g., based on softmax routing or Gaussian blurring) needs to be implemented for the physics loss to be meaningful.
*   **PDE Residual Calculation**: The calculation of the physics tendency (`dhdt_physics`) within `compute_pde_residual` in `src/losses.py` assumes the necessary spatial derivatives and drainage area can be computed from the PINN output `h_pred`. This needs careful implementation depending on whether `h_pred` represents scattered points or a grid, and relies on the differentiable drainage area function.
*   **Model `predict_state` Mode**: The `forward` method in `src/models.py` has a placeholder for the `predict_state` mode. This needs to be implemented based on how the model should predict the final grid state given the inputs from the data loader (e.g., initial conditions, parameters).
*   **Optimization Objective**: The `create_optimization_objective` function in `src/optimizer_utils.py` currently assumes the PINN's output is implicitly dependent on the parameter being optimized. For optimizing PDE parameters like `uplift_rate`, `K_f`, or `K_d` that were *fixed* during training, the PINN architecture or the optimization objective might need significant redesign (e.g., making the parameter an input to the PINN, or making the parameter itself network weights).
*   **Fastscape Compilation/Call**: Ensure the `fastscapelib-fortran` library compiles correctly on the target system and that the `xarray-simlab` calls in `scripts/generate_data.py` accurately reflect the required parameters and API usage for the `basic_model`.
*   **Normalization**: Implement proper calculation and saving/loading of normalization statistics based on the generated training data. Update `configs/train_config.yaml` accordingly.

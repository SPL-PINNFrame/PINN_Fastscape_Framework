import argparse
import logging
import os
import sys
import numpy as np
import torch
import time # Import time for run_name default

# Add src directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.utils import load_config, setup_logging, get_device
# Import all potential model types
from src.models import FastscapePINN, AdaptiveFastscapePINN, MLP_PINN
# Import the new PyTorch-based optimizer utility
from src.optimizer_utils import optimize_parameters # Keep only the new function
# Import interpolation function if needed for initial guess preparation
from src.optimizer_utils import interpolate_uplift_torch

# --- Helper functions for loading data ---

def load_target_dem(filepath, device):
    """Loads the target DEM and returns a torch tensor."""
    logging.info(f"Loading target DEM from {filepath}.")
    try:
        # Assume .npy format for simplicity, adjust if needed (e.g., GeoTIFF using rasterio)
        target_dem_np = np.load(filepath)
        target_dem_torch = torch.from_numpy(target_dem_np).float().unsqueeze(0).unsqueeze(0).to(device) # Add B, C dims
        logging.info(f"Target DEM loaded with shape: {target_dem_torch.shape}")
        return target_dem_torch
    except FileNotFoundError:
        logging.error(f"Target DEM file not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading target DEM from {filepath}: {e}")
        sys.exit(1)


def load_fixed_inputs(config, target_shape, device, dtype):
    """Loads fixed inputs specified in the config (e.g., initial_topo, K, D)."""
    fixed_input_paths = config.get('fixed_inputs', {})
    inputs = {}
    logging.info("Loading fixed inputs...")
    target_h, target_w = target_shape # H, W

    # Load initial topography
    init_topo_path = fixed_input_paths.get('initial_topography')
    if init_topo_path and os.path.exists(init_topo_path):
        try:
            init_topo_np = np.load(init_topo_path) # Assume .npy
            # Ensure shape matches target, add B, C dims
            if init_topo_np.shape != target_shape:
                 logging.warning(f"Initial topo shape {init_topo_np.shape} differs from target {target_shape}. Resizing.")
                 # Example resize using numpy/scipy or torch interpolation before converting
                 # For simplicity, let's assume it should match or error out for now
                 raise ValueError("Initial topo shape must match target DEM shape.")
            inputs['initial_topography'] = torch.from_numpy(init_topo_np).to(dtype).unsqueeze(0).unsqueeze(0).to(device)
            logging.info(f"Loaded initial topography from {init_topo_path}")
        except Exception as e:
            logging.error(f"Error loading initial topography from {init_topo_path}: {e}. Using zeros.")
            inputs['initial_topography'] = torch.zeros(1, 1, target_h, target_w, device=device, dtype=dtype)
    else:
        logging.warning("Initial topography path not found or specified. Using zeros.")
        inputs['initial_topography'] = torch.zeros(1, 1, target_h, target_w, device=device, dtype=dtype)

    # Load other fixed parameters (e.g., K, D) - these might be scalar or spatial
    # The ParameterOptimizer handles conversion of scalars to tensors later
    fixed_params_config = config.get('fixed_parameters', {}) # Look for fixed params here
    for key, value in fixed_params_config.items():
         if isinstance(value, str) and os.path.exists(value): # Check if value is a path
              try:
                   param_np = np.load(value)
                   # Ensure shape matches target, add B, C dims
                   if param_np.shape != target_shape:
                        raise ValueError(f"Fixed parameter '{key}' shape {param_np.shape} must match target {target_shape}.")
                   inputs[key] = torch.from_numpy(param_np).to(dtype).unsqueeze(0).unsqueeze(0).to(device)
                   logging.info(f"Loaded fixed parameter '{key}' from {value}")
              except Exception as e:
                   logging.error(f"Error loading fixed parameter '{key}' from {value}: {e}. Skipping.")
         else:
              # Assume scalar value
              try:
                   inputs[key] = float(value) # Store scalar, ParameterOptimizer will handle tensor conversion
                   logging.info(f"Using scalar value {inputs[key]} for fixed parameter '{key}'.")
              except ValueError:
                   logging.error(f"Invalid scalar value for fixed parameter '{key}': {value}. Skipping.")

    return inputs


def prepare_initial_param_guess(config, target_shape, device, dtype):
    """Prepares the initial guess for the parameters being optimized."""
    params_to_optimize_config = config.get('parameters_to_optimize', {})
    initial_params = {}

    for name, p_config in params_to_optimize_config.items():
        initial_guess_type = p_config.get('initial_guess_type', 'constant')
        initial_guess_value = p_config.get('initial_value', 0.0) # Default to 0 if not specified
        param_shape_config = p_config.get('parameter_shape', None) # Optional low-res shape

        if param_shape_config: # Low-resolution parameterization
             param_shape = tuple(param_shape_config)
             logging.info(f"Initializing low-resolution parameter '{name}' with shape {param_shape}.")
             if initial_guess_type == 'constant':
                  low_res_guess = torch.full(param_shape, float(initial_guess_value), device=device, dtype=dtype)
             elif initial_guess_type == 'random':
                  low = p_config.get('initial_random_low', 0.0)
                  high = p_config.get('initial_random_high', 1.0) # Adjust default range if needed
                  low_res_guess = torch.rand(param_shape, device=device, dtype=dtype) * (high - low) + low
             # Add loading from file if needed
             else:
                  raise ValueError(f"Unsupported initial_guess_type for low-res param '{name}': {initial_guess_type}")

             # Interpolate low-res guess to target shape
             # Use torch interpolation (ensure interpolate_uplift_torch exists and works)
             try:
                  # Add batch/channel dims for interpolation function if needed
                  # Assuming interpolate_uplift_torch handles [H,W] -> [H_tgt, W_tgt]
                  # Or adapt it to handle [B,1,H,W] -> [B,1,H_tgt,W_tgt]
                  # For now, assume it works on 2D grids directly
                  initial_params[name] = interpolate_uplift_torch(
                       low_res_guess.flatten(), # Pass flattened low-res tensor
                       param_shape,
                       target_shape,
                       method='bilinear' # Or method from config
                  ).unsqueeze(0).unsqueeze(0) # Add B, C dims
                  logging.info(f"Interpolated initial guess for '{name}' to shape {initial_params[name].shape}")
             except NameError:
                  logging.error("interpolate_uplift_torch function not found. Cannot interpolate low-res guess.")
                  # Fallback: Use constant value at target resolution
                  initial_params[name] = torch.full((1, 1, *target_shape), float(initial_guess_value), device=device, dtype=dtype)

        else: # Full-resolution parameterization
             logging.info(f"Initializing full-resolution parameter '{name}'.")
             target_full_shape = (1, 1, *target_shape) # B, C, H, W
             if initial_guess_type == 'constant':
                  initial_params[name] = torch.full(target_full_shape, float(initial_guess_value), device=device, dtype=dtype)
             elif initial_guess_type == 'random':
                  low = p_config.get('initial_random_low', 0.0)
                  high = p_config.get('initial_random_high', 1.0)
                  initial_params[name] = torch.rand(target_full_shape, device=device, dtype=dtype) * (high - low) + low
             # Add loading from file if needed
             else:
                  raise ValueError(f"Unsupported initial_guess_type for full-res param '{name}': {initial_guess_type}")

        # Ensure requires_grad is set
        initial_params[name].requires_grad_(True)

    return initial_params


def main(args):
    """Main function to run parameter optimization using PyTorch optimizers."""
    config = load_config(args.config)
    opt_config = config.get('optimization', {})
    model_config = config.get('model', {})
    run_config = config.get('run_options', {}) # General run options

    # --- Setup Logging ---
    output_dir = run_config.get('output_dir', 'results/')
    run_name = opt_config.get('run_name', f'optimize_run_{int(time.time())}')
    log_dir = os.path.join(output_dir, run_name, 'logs')
    opt_output_dir = os.path.join(output_dir, run_name, 'optimize_output') # Specific dir for opt results
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(opt_output_dir, exist_ok=True)
    setup_logging(log_dir=log_dir)

    device = get_device(config)
    # Determine model dtype for consistency
    model_dtype_str = model_config.get('dtype', 'float32')
    model_dtype = torch.float32 if model_dtype_str == 'float32' else torch.float64
    logging.info(f"Using device: {device}, Model dtype: {model_dtype}")

    # --- Load trained PINN model ---
    # Determine model class based on config or type saved in checkpoint
    # For now, assume model type is specified in config
    model_type_name = model_config.get('name', 'FastscapePINN') # Default to FastscapePINN
    if model_type_name == 'AdaptiveFastscapePINN':
         model_class = AdaptiveFastscapePINN
    elif model_type_name == 'FastscapePINN':
         model_class = FastscapePINN
    elif model_type_name == 'MLP_PINN':
         model_class = MLP_PINN
    else:
         logging.error(f"Unknown model name '{model_type_name}' in config.")
         sys.exit(1)

    # Instantiate model with config, excluding 'name' and 'dtype' if they are not constructor args
    model_args = {k: v for k, v in model_config.items() if k not in ['name', 'dtype']}
    model = model_class(**model_args).to(dtype=model_dtype) # Instantiate with specific dtype

    checkpoint_path = opt_config.get('pinn_checkpoint_path')
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logging.error(f"PINN checkpoint path not found or specified: {checkpoint_path}")
        sys.exit(1)
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu') # Load to CPU first
        # Load state dict, handling potential dtype mismatches if necessary
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device) # Move model to target device
        model.eval() # Set to evaluation mode
        logging.info(f"Loaded trained PINN model ({model_class.__name__}) from {checkpoint_path}")
        # Log model parameter count
        num_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Model parameter count: {num_params:,}")
    except Exception as e:
        logging.error(f"Error loading model checkpoint: {e}", exc_info=True)
        sys.exit(1)

    # --- Load target DEM ---
    target_dem_path = opt_config.get('target_dem_path')
    if not target_dem_path:
         logging.error("Target DEM path ('target_dem_path') not specified in optimization config.")
         sys.exit(1)
    target_dem_tensor = load_target_dem(target_dem_path, device).to(dtype=model_dtype) # Match model dtype
    target_dem_shape = target_dem_tensor.shape[2:] # H, W

    # --- Load fixed model inputs ---
    # Pass target_shape and dtype for consistency
    fixed_inputs = load_fixed_inputs(opt_config, target_dem_shape, device, model_dtype)

    # --- Prepare initial guess for parameters to optimize ---
    params_to_optimize_config = opt_config.get('parameters_to_optimize')
    if not params_to_optimize_config:
         logging.error("'parameters_to_optimize' section not found in optimization config.")
         sys.exit(1)
    initial_params_guess = prepare_initial_param_guess(opt_config, target_dem_shape, device, model_dtype)


    # --- Get Target Time ---
    # Target time should correspond to the observation time
    t_target_value = opt_config.get('t_target', None)
    if t_target_value is None:
         # Try getting from physics_params if defined globally
         t_target_value = config.get('physics_params', {}).get('total_time')
         if t_target_value is None:
              logging.error("Target time 't_target' not specified in optimization config or physics_params.")
              sys.exit(1)
    logging.info(f"Using target time: {t_target_value}")


    # --- Run Optimization using the new utility function ---
    # Pass the main config dict which contains 'optimization_params'
    # Ensure the save path is correctly configured within the main config
    opt_config['optimization_params']['save_path'] = os.path.join(opt_output_dir, opt_config.get('output_filename', 'optimized_params.pth'))

    optimized_params, history = optimize_parameters(
        model=model,
        observation_data=target_dem_tensor,
        params_to_optimize_config=params_to_optimize_config,
        config=config, # Pass the main config
        initial_state=fixed_inputs.get('initial_topography'),
        fixed_params={k: v for k, v in fixed_inputs.items() if k != 'initial_topography'}, # Pass other fixed inputs
        t_target=t_target_value
    )

    # --- Post-processing and Saving ---
    logging.info("Optimization process completed.")
    if history['loss']:
         logging.info(f"Final loss: {history['final_loss']:.6e}")
    else:
         logging.warning("Optimization history is empty.")

    # Optimized parameters are already saved by optimize_parameters if save_path is set
    # You can add further analysis or visualization here if needed
    # Example: Compare optimized U with initial guess or known true value if available
    if 'U' in optimized_params:
         logging.info(f"Optimized U parameter mean: {optimized_params['U'].mean().item():.6e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize landscape evolution parameters using a trained PINN.")
    parser.add_argument('--config', type=str, required=True, help='Path to the optimization configuration file.')
    args = parser.parse_args()
    main(args)

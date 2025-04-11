import argparse
import logging
import os
import sys
import numpy as np
import torch
import xsimlab as xs
import fastscape # Import the high-level package
import math # For spatial field generation

# Add src directory to Python path to allow importing modules from src
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.utils import load_config, setup_logging
# from src.data_utils import FastscapeWrapper # Removed, use xsimlab/fastscape directly
from fastscape.models import basic_model # Example: Import a specific model

# --- Helper function for spatial parameter generation ---
def generate_spatial_field(shape, min_val, max_val, pattern='random'):
    """Generates a spatial parameter field with a specified pattern."""
    logging.debug(f"Generating spatial field: shape={shape}, min={min_val}, max={max_val}, pattern='{pattern}'")
    if pattern == 'random':
        field = np.random.uniform(min_val, max_val, shape)
    elif pattern == 'fault':
        field = np.ones(shape) * min_val
        fault_pos_rel = np.random.uniform(0.3, 0.7)
        fault_angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        center_y = shape[0] / 2
        center_x = shape[1] / 2
        y_indices, x_indices = np.indices(shape)
        y_rel = y_indices - center_y
        x_rel = x_indices - center_x
        rotated_y = y_rel * np.cos(fault_angle) - x_rel * np.sin(fault_angle)
        split_value = (shape[0] * fault_pos_rel) - center_y
        field[rotated_y > split_value] = max_val
        logging.debug(f"Generated fault pattern: angle={fault_angle:.2f}, pos={fault_pos_rel:.2f}")
    elif pattern == 'constant':
         field = np.full(shape, min_val)
         logging.debug(f"Generated constant field with value: {min_val}")
    else:
        logging.warning(f"Unknown spatial pattern: '{pattern}'. Using random uniform.")
        field = np.random.uniform(min_val, max_val, shape)
    return field.astype(np.float32)

# --- Function for sampling SCALAR parameters ---
def sample_scalar_parameters(param_ranges):
    """Samples SCALAR parameters for a Fastscape simulation run."""
    sampled_params = {}
    # Define parameters that are ALWAYS scalar or sampled as scalar
    # Note: uplift, k_coef, diffusivity might be overwritten if spatial mode is active
    scalar_params_config = {
        'uplift__rate': [1e-4, 1e-3],
        'spl__k_coef': [1e-6, 1e-5],
        'diffusion__diffusivity': [0.1, 1.0],
        'spl__area_exp': [0.4, 0.6], # m is typically scalar
        'spl__slope_exp': [1.0, 1.0]  # n is often fixed or scalar
    }
    for key, default_range in scalar_params_config.items():
        current_range = param_ranges.get(key, default_range)
        if isinstance(current_range, list) and len(current_range) == 2:
             if current_range[0] == current_range[1]:
                  sampled_params[key] = float(current_range[0])
             else:
                  sampled_params[key] = np.random.uniform(*current_range)
        elif isinstance(current_range, (int, float)):
             sampled_params[key] = float(current_range)
        else:
             logging.warning(f"Invalid range format for scalar parameter '{key}': {current_range}. Using default range {default_range}.")
             sampled_params[key] = np.random.uniform(*default_range)

    logging.info(f"Sampled SCALAR parameters (initial): {sampled_params}")
    return sampled_params

# --- Function for generating potentially SPATIAL parameters ---
def generate_potentially_spatial_parameters(spatial_config, grid_shape):
    """Generates parameters which CAN be spatial, based on config."""
    generated_params = {}
    # Define parameters that CAN be spatial (as per user request)
    spatial_capable_params = ['uplift__rate', 'spl__k_coef'] # Only U and K_f are spatial

    for param_key in spatial_capable_params:
        config = spatial_config.get(param_key, {})
        pattern = config.get('pattern', 'constant') # Default to constant if pattern not specified
        min_val = config.get('min', 0.0)
        max_val = config.get('max', 1.0)

        try:
            min_val = float(min_val)
            max_val = float(max_val)
        except ValueError:
            logging.error(f"Invalid min/max value for {param_key}: min={min_val}, max={max_val}. Using defaults.")
            min_val, max_val = 0.0, 1.0

        # Always generate spatial field if config exists for this key, even if pattern is 'constant'
        # Let the generate_spatial_field handle the 'constant' pattern.
        generated_params[param_key] = generate_spatial_field(grid_shape, min_val, max_val, pattern)

    logging.info(f"Generated potentially SPATIAL parameters: {list(generated_params.keys())}")
    return generated_params


def save_sample(sample_data, output_dir, sample_index):
    """Saves a generated sample (dictionary of tensors and numpy arrays)."""
    filename = f"sample_{sample_index:05d}.pt"
    filepath = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True) # Ensure directory exists before saving
    try:
        torch.save(sample_data, filepath)
        logging.debug(f"Saved sample to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save sample {sample_index} to {filepath}: {e}")

def generate_single_sample(sample_index, config):
    """Generates and saves a single simulation sample based on the provided config."""
    data_gen_config = config.get('data_generation', {})
    sim_config = data_gen_config.get('simulation_params', {})
    parameter_type = data_gen_config.get('parameter_type', 'scalar').lower()
    scalar_param_ranges = data_gen_config.get('parameter_ranges', {})
    spatial_param_config = data_gen_config.get('spatial_parameter_config', {})
    output_dir = data_gen_config.get('output_dir') # Expect specific output dir in config

    # Get simulation parameters for this specific sample/resolution
    grid_shape = tuple(sim_config.get('grid_shape'))
    grid_length = sim_config.get('grid_length')
    time_step = sim_config.get('time_step')
    run_time_total = sim_config.get('run_time')

    logging.info(f"--- Generating sample {sample_index+1} for shape {grid_shape} ---")
    try:
        # 1. Generate/Sample Parameters
        final_params = sample_scalar_parameters(scalar_param_ranges)
        if parameter_type == 'spatial':
            spatial_params_to_generate = ['uplift__rate', 'spl__k_coef']
            spatial_params = {}
            for key in spatial_params_to_generate:
                 if key in spatial_param_config:
                      config_for_key = spatial_param_config[key]
                      pattern = config_for_key.get('pattern', 'constant')
                      min_val = float(config_for_key.get('min', 0.0))
                      max_val = float(config_for_key.get('max', 1.0))
                      # Use the specific grid_shape for this sample
                      spatial_params[key] = generate_spatial_field(grid_shape, min_val, max_val, pattern)
                 else:
                      logging.warning(f"Config for spatial parameter '{key}' not found. It will remain scalar.")
            final_params.update(spatial_params)

        logging.debug(f"Final parameters for sample {sample_index+1}: { {k: type(v) for k, v in final_params.items()} }")

        # 2. Setup xsimlab Simulation
        sim_times = np.arange(0, run_time_total + time_step, time_step)
        output_times = [0, run_time_total]
        model_to_use = basic_model

        input_vars_dict = {
            'grid__shape': list(grid_shape),
            'grid__length': grid_length,
            'boundary__status': sim_config.get('boundary_status', 'fixed_value'),
        }
        for key, value in final_params.items():
            if isinstance(value, np.ndarray) and value.ndim == 2:
                if key in ['spl__area_exp', 'spl__slope_exp']:
                     logging.error(f"Parameter '{key}' generated as spatial but must be scalar. Using mean.")
                     input_vars_dict[key] = float(value.mean())
                else:
                     input_vars_dict[key] = (('y', 'x'), value)
            else:
                input_vars_dict[key] = value

        output_vars_dict = {'topography__elevation': 'out'}

        in_ds = xs.create_setup(
            model=model_to_use,
            clocks={'time': sim_times, 'out': output_times},
            master_clock='time',
            input_vars=input_vars_dict,
            output_vars=output_vars_dict
        )

        # 3. Run Simulation
        logging.info(f"Sample {sample_index+1}: Starting xsimlab.run...")
        out_ds = in_ds.xsimlab.run(model=model_to_use)
        logging.info(f"Sample {sample_index+1}: xsimlab.run finished.")

        # 4. Extract and Format Data
        if len(out_ds['out']) < 2 or out_ds['out'][0] != 0 or out_ds['out'][-1] != run_time_total:
             logging.warning(f"Output times {out_ds['out'].values} mismatch. Adjusting extraction.")
             initial_topo_xr = out_ds['topography__elevation'].sel(out=0, method='nearest')
             final_topo_xr = out_ds['topography__elevation'].sel(out=run_time_total, method='nearest')
        else:
             initial_topo_xr = out_ds['topography__elevation'].isel(out=0)
             final_topo_xr = out_ds['topography__elevation'].isel(out=-1)

        initial_topo_tensor = torch.tensor(initial_topo_xr.values, dtype=torch.float32).unsqueeze(0)
        final_topo_tensor = torch.tensor(final_topo_xr.values, dtype=torch.float32).unsqueeze(0)

        sample_output = {
            'initial_topo': initial_topo_tensor,
            'final_topo': final_topo_tensor,
            'run_time': torch.tensor(run_time_total, dtype=torch.float32)
        }
        param_mapping = {
            'uplift__rate': 'uplift_rate', 'spl__k_coef': 'k_f',
            'diffusion__diffusivity': 'k_d', 'spl__area_exp': 'm', 'spl__slope_exp': 'n'
        }
        for xsim_key, data_key in param_mapping.items():
            if xsim_key in final_params:
                value = final_params[xsim_key]
                if isinstance(value, np.ndarray):
                    sample_output[data_key] = value
                else:
                    sample_output[data_key] = torch.tensor(value, dtype=torch.float32)

        # 5. Save Sample
        save_sample(sample_output, output_dir, sample_index)
        logging.info(f"Sample {sample_index+1}: Successfully saved.")
        return True # Indicate success

    except Exception as e:
        logging.error(f"Failed to generate or save sample {sample_index+1}: {e}", exc_info=True)
        return False # Indicate failure

def generate_dataset(config):
    """Generates a dataset for a specific configuration (resolution)."""
    data_gen_config = config.get('data_generation', {})
    num_samples = data_gen_config.get('num_samples', 10)
    output_dir = data_gen_config.get('output_dir')
    grid_shape = tuple(data_gen_config.get('simulation_params', {}).get('grid_shape', [0,0]))

    logging.info(f"Generating dataset: {num_samples} samples for shape {grid_shape} into {output_dir}")
    # Explicitly create the output directory for this resolution here
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.debug(f"Ensured output directory exists: {output_dir}")
    except OSError as e:
        logging.error(f"Failed to create output directory {output_dir}: {e}")
        # Decide how to handle this - maybe skip this resolution?
        # For now, log the error and continue; the sample saving will likely fail later.
        pass

    generated_count = 0
    for i in range(num_samples):
        if generate_single_sample(i, config):
            generated_count += 1

    logging.info(f"Dataset generation finished for shape {grid_shape}. {generated_count}/{num_samples} samples generated.")


def main(args):
    """Main function to generate data using xarray-simlab and fastscape, potentially at multiple scales."""
    # --- Setup ---
    config = load_config(args.config)
    log_config = config.get('logging', {})
    log_dir = log_config.get('log_dir', 'logs/data_generation')
    log_filename = log_config.get('log_filename', 'generate_data.log')
    log_file_path = os.path.join(log_dir, log_filename) if log_dir and log_filename else None
    # Get log level from config, default to INFO
    log_level = log_config.get('log_level', 'INFO')
    setup_logging(log_level=log_level, log_file=log_file_path, log_to_console=True) # Pass the full path

    base_data_gen_config = config.get('data_generation', {})
    base_output_dir = base_data_gen_config.get('output_dir', 'data/processed')

    # --- Multi-scale Generation Setup ---
    # Define resolutions to generate (can be moved to config)
    resolutions = base_data_gen_config.get('resolutions', [(64, 64)]) # Default to single scale if not specified
    domain_size_x = base_data_gen_config.get('domain_size_x', 10000.0) # Physical domain size X
    domain_size_y = base_data_gen_config.get('domain_size_y', 10000.0) # Physical domain size Y

    logging.info(f"Starting multi-scale data generation for resolutions: {resolutions}")
    logging.info(f"Base output directory: {base_output_dir}")
    logging.info(f"Physical domain size: {domain_size_x} x {domain_size_y}")

    for height, width in resolutions:
        logging.info(f"--- Processing resolution: {height}x{width} ---")

        # Create a specific config for this resolution
        res_config = config.copy() # Start with a copy of the full config
        res_data_gen_config = res_config['data_generation'].copy() # Copy the data_gen part

        # Update simulation parameters for this resolution
        sim_params = res_data_gen_config.get('simulation_params', {}).copy()
        sim_params['grid_shape'] = (height, width)
        # Keep grid_length as the physical domain size
        sim_params['grid_length'] = [domain_size_y, domain_size_x]
        # dx and dy are implicitly defined by grid_shape and grid_length in fastscape
        res_data_gen_config['simulation_params'] = sim_params

        # Define and create output directory for this resolution
        res_output_dir = os.path.join(base_output_dir, f"resolution_{height}x{width}")
        os.makedirs(res_output_dir, exist_ok=True)
        res_data_gen_config['output_dir'] = res_output_dir

        # Update the config dictionary passed to the generation function
        res_config['data_generation'] = res_data_gen_config

        # Generate the dataset for this resolution
        generate_dataset(res_config) # Pass the modified config

    logging.info("Multi-scale data generation finished.")


# --- Original single-scale generation logic (now moved into generate_dataset/generate_single_sample) ---
#    os.makedirs(output_dir, exist_ok=True)
#    logging.info(f"Starting data generation. Number of samples: {num_samples}")
#    logging.info(f"Output directory: {output_dir}")
#    logging.info(f"Parameter generation mode: {parameter_type}")
#    logging.info(f"Simulation base config: {sim_config}")
#
#    # --- Parameter Config Handling ---
#    scalar_param_ranges = data_gen_config.get('parameter_ranges', {})
#    logging.info(f"Scalar parameter ranges/values: {scalar_param_ranges}")
#
#    spatial_param_config = {}
#    if parameter_type == 'spatial':
#        spatial_param_config = data_gen_config.get('spatial_parameter_config', {})
#        logging.info(f"Spatial parameter config: {spatial_param_config}")
#        if not spatial_param_config:
#             logging.warning("Parameter type is 'spatial' but 'spatial_parameter_config' is missing or empty.")
#
#
#    # --- Generation Loop ---
#    generated_count = 0
#    for i in range(num_samples):
#        logging.info(f"--- Generating sample {i+1}/{num_samples} ---")
#        try:
#            # 1. Generate/Sample Parameters
#            # Start with scalar parameters (includes m, n, and potentially U, K_f, K_d if not spatial)
#            final_params = sample_scalar_parameters(scalar_param_ranges)
#
#            # If mode is spatial, overwrite U and K_f with spatial fields
#            if parameter_type == 'spatial':
#                # Generate only U and K_f spatially
#                spatial_params_to_generate = ['uplift__rate', 'spl__k_coef']
#                spatial_params = {}
#                for key in spatial_params_to_generate:
#                     if key in spatial_param_config:
#                          config_for_key = spatial_param_config[key]
#                          pattern = config_for_key.get('pattern', 'constant')
#                          min_val = float(config_for_key.get('min', 0.0))
#                          max_val = float(config_for_key.get('max', 1.0))
#                          spatial_params[key] = generate_spatial_field(grid_shape, min_val, max_val, pattern)
#                     else:
#                          logging.warning(f"Config for spatial parameter '{key}' not found in 'spatial_parameter_config'. It will remain scalar.")
#
#                final_params.update(spatial_params) # Overwrite U and K_f if generated spatially
#
#            logging.debug(f"Final parameters for sample {i+1}: { {k: type(v) for k, v in final_params.items()} }")
#
#            # 2. Setup xsimlab Simulation
#            sim_times = np.arange(0, run_time_total + time_step, time_step)
#            output_times = [0, run_time_total]
#            model_to_use = basic_model
#
#            input_vars_dict = {
#                'grid__shape': list(grid_shape),
#                'grid__length': grid_length,
#                'boundary__status': sim_config.get('boundary_status', 'fixed_value'),
#            }
#            # Add generated parameters, specifying dims for spatial arrays
#            for key, value in final_params.items():
#                if isinstance(value, np.ndarray) and value.ndim == 2:
#                    # Check if this parameter is allowed to be spatial by fastscape model
#                    # Based on previous error, 'spl__area_exp' and 'spl__slope_exp' must be scalar
#                    if key in ['spl__area_exp', 'spl__slope_exp']:
#                         logging.error(f"Parameter '{key}' was generated as spatial but must be scalar for the model. Using mean value.")
#                         input_vars_dict[key] = float(value.mean())
#                    else:
#                         input_vars_dict[key] = (('y', 'x'), value)
#                else: # Assume scalar
#                    input_vars_dict[key] = value
#
#            output_vars_dict = {'topography__elevation': 'out'}
#
#            in_ds = xs.create_setup(
#                model=model_to_use,
#                clocks={'time': sim_times, 'out': output_times},
#                master_clock='time',
#                input_vars=input_vars_dict,
#                output_vars=output_vars_dict
#            )
#
#            # 3. Run Simulation
#            logging.info(f"Sample {i+1}: Setting up xsimlab model...")
#            logging.info(f"Sample {i+1}: Starting xsimlab.run...")
#            out_ds = in_ds.xsimlab.run(model=model_to_use)
#            logging.info(f"Sample {i+1}: xsimlab.run returned dataset: {out_ds}")
#            logging.info(f"Sample {i+1}: xsimlab.run finished.")
#            print(f"DEBUG PRINT: Sample {i+1}: xsimlab.run finished. out_ds info: {out_ds}")
#
#            # 4. Extract and Format Data
#            logging.info(f"Sample {i+1}: Extracting data from output dataset...")
#            logging.debug(f"Sample {i+1}: Output dataset keys: {list(out_ds.keys())}")
#            if len(out_ds['out']) < 2 or out_ds['out'][0] != 0 or out_ds['out'][-1] != run_time_total:
#                 logging.warning(f"Output times {out_ds['out'].values} do not match expected [0, {run_time_total}]. Adjusting extraction.")
#                 initial_topo_xr = out_ds['topography__elevation'].sel(out=0, method='nearest')
#                 final_topo_xr = out_ds['topography__elevation'].sel(out=run_time_total, method='nearest')
#            else:
#                 initial_topo_xr = out_ds['topography__elevation'].isel(out=0)
#                 final_topo_xr = out_ds['topography__elevation'].isel(out=-1)
#
#            logging.debug(f"Sample {i+1}: Extracted initial_topo_xr: {initial_topo_xr.shape}, Extracted final_topo_xr: {final_topo_xr.shape}")
#            print(f"DEBUG PRINT: Sample {i+1}: Extracted shapes: initial={initial_topo_xr.shape}, final={final_topo_xr.shape}")
#
#            initial_topo_tensor = torch.tensor(initial_topo_xr.values, dtype=torch.float32).unsqueeze(0)
#            final_topo_tensor = torch.tensor(final_topo_xr.values, dtype=torch.float32).unsqueeze(0)
#
#            # Prepare dictionary for saving
#            sample_output = {
#                'initial_topo': initial_topo_tensor,
#                'final_topo': final_topo_tensor,
#                'run_time': torch.tensor(run_time_total, dtype=torch.float32)
#            }
#            # Add generated parameters (map keys and convert types)
#            param_mapping = {
#                'uplift__rate': 'uplift_rate',
#                'spl__k_coef': 'k_f',
#                'diffusion__diffusivity': 'k_d',
#                'spl__area_exp': 'm',
#                'spl__slope_exp': 'n'
#            }
#            for xsim_key, data_key in param_mapping.items():
#                if xsim_key in final_params:
#                    value = final_params[xsim_key]
#                    if isinstance(value, np.ndarray): # Spatial field
#                        sample_output[data_key] = value # Save as numpy array
#                    else: # Scalar
#                        sample_output[data_key] = torch.tensor(value, dtype=torch.float32) # Save as 0-dim tensor
#
#            print(f"DEBUG PRINT: Sample {i+1}: Saving data with keys: {list(sample_output.keys())}")
#            # 5. Save Sample
#            logging.info(f"Sample {i+1}: Preparing to save sample {i}...")
#            logging.debug(f"Sample {i+1}: Data prepared for saving: keys={list(sample_output.keys())}")
#            save_sample(sample_output, output_dir, i)
#            logging.info(f"Sample {i+1}: Successfully saved sample {i}.")
#            generated_count += 1
#
#        except Exception as e:
#            logging.error(f"Failed to generate or save sample {i+1}: {e}", exc_info=True)
#
#    logging.info(f"Data generation finished. {generated_count}/{num_samples} samples successfully generated.")

# --- (Removed redundant single-scale logic previously here) ---
# --- (Removed redundant generation loop previously here) ---
# The main generation logic is now handled within generate_dataset -> generate_single_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data using Fastscape.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    main(args)

import logging
import os
import sys
import torch
import numpy as np

# Add project root directory (one level up from PINN_Fastscape_Framework) to Python path
# This allows imports like 'from PINN_Fastscape_Framework.src import ...'
script_dir = os.path.dirname(os.path.abspath(__file__))
pinn_framework_dir = os.path.dirname(script_dir) # This is PINN_Fastscape_Framework
project_root = os.path.dirname(pinn_framework_dir) # This is the main project root
# Add project root to the beginning of the path to prioritize it
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now use absolute imports from the project root
try:
    from PINN_Fastscape_Framework.src.physics import validate_drainage_area
    from PINN_Fastscape_Framework.src.utils import setup_logging # For consistent logging setup
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure the script is run from the project root or the environment has the correct paths.")
    sys.exit(1)
    print("Ensure the script is run from the project root or the environment has the correct paths.")
    sys.exit(1)

def create_test_topography(height=65, width=65, slope=0.01):
    """Creates a simple tilted plane topography."""
    y_coords = torch.linspace(0, height - 1, height)
    x_coords = torch.linspace(0, width - 1, width)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    # Create a slope primarily in the y-direction
    topo = 100.0 - yy * slope
    # Add batch and channel dimensions
    return topo.unsqueeze(0).unsqueeze(0).float()

def main():
    """Runs the drainage area validation test."""
    # Setup logging
    # Use the project_root defined earlier for log path
    log_dir = os.path.join(project_root, 'PINN_Fastscape_Framework', 'logs', 'component_tests')
    setup_logging(log_dir=log_dir, log_filename="test_drainage_validation.log")
    logging.info("Starting drainage area validation test...")

    # --- Test Parameters ---
    height, width = 65, 65
    dx, dy = 100.0, 100.0 # Match default grid spacing if possible
    pinn_params = {'temp': 0.01, 'num_iters': 50} # Default params from function

    # --- Create Test Data ---
    logging.info(f"Creating test topography ({height}x{width})...")
    test_h = create_test_topography(height, width)
    logging.info(f"Test topography created with shape: {test_h.shape}")

    # --- Run Validation ---
    logging.info("Running validate_drainage_area...")

    try:
        validation_results = validate_drainage_area(
            test_h, dx, dy, pinn_method_params=pinn_params, d8_method='fastscape'
        )
        logging.info("Validation finished.")
        logging.info("Validation Results:")
        for key, value in validation_results.items():
            # Check for NaN before formatting
            if isinstance(value, float) and np.isnan(value):
                 log_value = "NaN"
            else:
                 # Format floats for better readability
                 log_value = f"{value:.4f}" if isinstance(value, float) else value
            logging.info(f"  - {key}: {log_value}")

    except ImportError as e:
         logging.error(f"ImportError during validation: {e}. Is 'fastscape' installed correctly in the environment?")
    except Exception as e:
        logging.error("An unexpected error occurred during validation:", exc_info=True)

if __name__ == "__main__":
    main()
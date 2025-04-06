import pytest
import torch # Add torch import
import subprocess
import os
import sys
import shutil
import yaml # Import yaml to read the config file
import yaml # Import yaml to read the config file
import pytest # Import pytest for markers
from unittest.mock import patch, MagicMock
# Ensure the project root is in the Python path so src can be imported by scripts
# Note: This might be needed depending on how pytest discovers/runs tests
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # Assumes tests/ is one level down from project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Test Configuration ---
# Paths relative to project_root (PINN_Fastscape_Framework)
CONFIG_FILE_REL = "configs/test_config_pytest.yaml"
# Paths relative to the pytest execution root directory (d:\...\code)
TEST_DATA_DIR = os.path.join("pytest_data", "processed") # Relative to project root (cwd)
TEST_RESULTS_DIR = "pytest_results" # Relative to project root (cwd)
# Script paths relative to project_root (PINN_Fastscape_Framework)
GENERATE_SCRIPT_REL = "scripts/generate_data.py"
TRAIN_SCRIPT_REL = "scripts/train.py"

# --- Fixture for Setup and Teardown ---
@pytest.fixture(scope="module") # Run setup/teardown once per module
def test_environment():
    """Cleans up test directories before and after the test."""
    print("\nSetting up test environment...")
    # Clean up before test (use the updated paths relative to pytest execution dir)
    data_base_dir = os.path.dirname(TEST_DATA_DIR) # e.g., PINN_Fastscape_Framework/pytest_data
    results_base_dir = TEST_RESULTS_DIR # e.g., PINN_Fastscape_Framework/pytest_results
    root_level_results_dir = "pytest_results" # The potentially problematic dir at root

    print(f"Attempting to remove: {data_base_dir}")
    if os.path.exists(data_base_dir):
        shutil.rmtree(data_base_dir)
    print(f"Attempting to remove: {results_base_dir}")
    if os.path.exists(results_base_dir):
        shutil.rmtree(results_base_dir)
    # Also attempt to remove the root level one if it exists from previous runs
    print(f"Attempting to remove root level: {root_level_results_dir}")
    if os.path.exists(root_level_results_dir):
         try:
              shutil.rmtree(root_level_results_dir)
              print(f"Successfully removed root level {root_level_results_dir}")
         except OSError as e:
              print(f"Could not remove root level {root_level_results_dir}: {e}")


    # Add assertions after cleaning
    assert not os.path.exists(data_base_dir), f"Directory {data_base_dir} still exists after cleanup!"
    assert not os.path.exists(results_base_dir), f"Directory {results_base_dir} still exists after cleanup!"
    assert not os.path.exists(root_level_results_dir), f"Root level directory {root_level_results_dir} still exists after cleanup!"
    print("Cleanup checks passed.")

    # Recreate the necessary directories relative to pytest execution dir
    print(f"Creating directory: {TEST_DATA_DIR}")
    os.makedirs(TEST_DATA_DIR, exist_ok=True) # Creates PINN_Fastscape_Framework/pytest_data/processed
    print(f"Creating directory: {TEST_RESULTS_DIR}")
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True) # Creates PINN_Fastscape_Framework/pytest_results

    # Add assertions after creation
    assert os.path.exists(TEST_DATA_DIR), f"Directory {TEST_DATA_DIR} was not created!"
    assert os.path.exists(TEST_RESULTS_DIR), f"Directory {TEST_RESULTS_DIR} was not created!"
    print("Directory creation checks passed.")

    yield # This is where the test runs

    print("\nTearing down test environment...")
    # Clean up after test
    # if os.path.exists(TEST_DATA_DIR):
    #     shutil.rmtree(TEST_DATA_DIR)
    # if os.path.exists(TEST_RESULTS_DIR):
    #     shutil.rmtree(TEST_RESULTS_DIR)
    # Keep the directories after test for inspection if needed, comment out cleanup

# --- The End-to-End Test ---
# Patch SummaryWriter to avoid TensorBoard dependency during test
@pytest.mark.dependency() # Mark this test as a dependency
@patch('src.trainer.SummaryWriter', MagicMock()) # Mock the class in the trainer module
def test_full_pipeline(test_environment):
    """
    Tests the full data generation and training pipeline with minimal settings.
    """
    # Construct absolute path for config file if needed by scripts, or keep relative if cwd works
    config_path_abs = os.path.join(project_root, CONFIG_FILE_REL)
    print(f"\nRunning E2E test with config: {config_path_abs}") # Log absolute path for clarity

    # --- Set Environment Variable for OMP ---
    env = os.environ.copy()
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    print("Set KMP_DUPLICATE_LIB_OK=TRUE")

    # --- Step 1: Run Data Generation ---
    # Use relative script path because cwd is set to project_root
    print(f"Running data generation script: {GENERATE_SCRIPT_REL}")
    # Pass relative config path as argument, assuming scripts handle paths relative to their location or cwd
    generate_cmd = [sys.executable, GENERATE_SCRIPT_REL, '--config', CONFIG_FILE_REL]
    generate_result = subprocess.run(generate_cmd, capture_output=True, text=True, env=env, cwd=project_root)

    print("\n--- generate_data.py STDOUT ---")
    print(generate_result.stdout)
    print("\n--- generate_data.py STDERR ---")
    print(generate_result.stderr)

    assert generate_result.returncode == 0, f"generate_data.py failed with exit code {generate_result.returncode}"
    print("generate_data.py completed successfully.")

    # Check if data files were created inside the resolution-specific subdirectory
    # --- Dynamically determine expected data directory ---
    # Read the resolution from the test config file
    try:
        with open(config_path_abs, 'r') as f:
            test_config = yaml.safe_load(f)
        # Assuming single resolution in the test config as per recent changes
        resolution = test_config['data_generation']['resolutions'][0]
        height, width = resolution
        expected_resolution_dir_name = f"resolution_{height}x{width}"
        expected_resolution_dir_path = os.path.join(TEST_DATA_DIR, expected_resolution_dir_name)
    except Exception as e:
        pytest.fail(f"Failed to read resolution from config file {config_path_abs}: {e}")

    assert os.path.exists(expected_resolution_dir_path), f"Test data subdirectory '{expected_resolution_dir_path}' was not created."
    data_files = [f for f in os.listdir(expected_resolution_dir_path) if f.endswith('.pt')]
    assert len(data_files) > 0, f"No .pt files found in {expected_resolution_dir_path} after generation."
    print(f"Found {len(data_files)} data files in {expected_resolution_dir_path}.")


    # --- Step 2: Run Training ---
    # Use relative script path
    print(f"\nRunning training script: {TRAIN_SCRIPT_REL}")
    # Pass relative config path
    train_cmd = [sys.executable, TRAIN_SCRIPT_REL, '--config', CONFIG_FILE_REL]
    # The patch is active here
    train_result = subprocess.run(train_cmd, capture_output=True, text=True, env=env, cwd=project_root)

    print("\n--- train.py STDOUT ---")
    print(train_result.stdout)
    print("\n--- train.py STDERR ---")
    print(train_result.stderr)

    # Check for ModuleNotFoundError specifically for tensorboard in stderr, even if return code is 0 initially
    # (The mock should prevent this, but as a safeguard)
    assert 'ModuleNotFoundError: No module named \'tensorboard\'' not in train_result.stderr, \
           "Tensorboard import error detected despite mock. Check mock target."

    assert train_result.returncode == 0, f"train.py failed with exit code {train_result.returncode}"
    print("train.py completed successfully.")

    # Check if results directory and some output (e.g., log file) were created
    assert os.path.exists(TEST_RESULTS_DIR), f"Test results directory '{TEST_RESULTS_DIR}' was not created."
    # Look for the specific run directory inside results
    run_name = "pinn_pytest_e2e_run" # Must match run_name in test_config_pytest.yaml
    run_dir = os.path.join(TEST_RESULTS_DIR, run_name)
    assert os.path.exists(run_dir), f"Specific run directory '{run_dir}' was not created."
    log_dir = os.path.join(run_dir, 'logs')
    assert os.path.exists(log_dir), f"Log directory '{log_dir}' was not created."
    # Check for a log file (assuming default name 'training.log' used by train.py's setup_logging)
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
    assert len(log_files) > 0, f"No log files found in {log_dir}."
    print(f"Found log files in {log_dir}.")

    print("\nE2E pipeline test passed!")

# To run this test:
# 1. Make sure pytest is installed (`pip install pytest pytest-mock`)
# 2. Navigate to the project root directory (`d:\OneDrive\MR.Z 所有资料\code`) in your terminal
# 3. Run the command: `pytest PINN_Fastscape_Framework/tests/test_e2e_pipeline.py -s -v`
#    (-s shows print statements, -v provides verbose output)


# --- Test Model Loading and Prediction ---
@pytest.mark.dependency(depends=["test_full_pipeline"]) # Specify dependency
@patch('src.trainer.SummaryWriter', MagicMock()) # Mock SummaryWriter if trainer is implicitly imported via model loading utils
def test_model_prediction(test_environment):
    """Tests loading a trained model checkpoint and making a prediction."""
    print("\nRunning model prediction test...")
    # --- Find the checkpoint ---
    run_name = "pinn_pytest_e2e_run" # Must match run_name in test_config_pytest.yaml
    checkpoint_dir = os.path.join(TEST_RESULTS_DIR, run_name, 'checkpoints')
    # Look for either best model or the epoch 0 checkpoint (since we only run 1 epoch)
    checkpoint_path_best = os.path.join(checkpoint_dir, 'best_model.pth')
    checkpoint_path_epoch = os.path.join(checkpoint_dir, 'epoch_0000.pth')

    if os.path.exists(checkpoint_path_best):
        checkpoint_path = checkpoint_path_best
    elif os.path.exists(checkpoint_path_epoch):
        checkpoint_path = checkpoint_path_epoch
    else:
        pytest.fail(f"No checkpoint file found in {checkpoint_dir}. Run test_full_pipeline first.")
    print(f"Found checkpoint: {checkpoint_path}")

    # --- Load Checkpoint and Config ---
    try:
        # Explicitly set weights_only=False to suppress FutureWarning,
        # as the checkpoint contains not only model weights but also the config dictionary.
        # TODO: Consider saving config separately to allow weights_only=True for model loading.
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config = checkpoint['config'] # Load config saved in checkpoint
        model_config = config.get('model', {})
        model_name = model_config.pop('name', 'FastscapePINN') # Default needed if name wasn't saved
        model_dtype_str = model_config.pop('dtype', 'float32')
        model_dtype = torch.float32 if model_dtype_str == 'float32' else torch.float64
        device = torch.device('cpu') # Force CPU for test prediction

        # Select model class
        if model_name == 'AdaptiveFastscapePINN':
            from src.models import AdaptiveFastscapePINN as ModelClass
        elif model_name == 'FastscapePINN':
            from src.models import FastscapePINN as ModelClass
        elif model_name == 'MLP_PINN':
            from src.models import MLP_PINN as ModelClass
        else:
            pytest.fail(f"Unknown model name '{model_name}' in loaded config.")

        # Instantiate model
        model_args = {k: v for k, v in model_config.items()} # Use remaining args
        model = ModelClass(**model_args).to(dtype=model_dtype)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"Model {model_name} loaded successfully from checkpoint.")
    except Exception as e:
        pytest.fail(f"Failed to load model from checkpoint {checkpoint_path}: {e}")

    # --- Prepare Dummy Input Data ---
    # Use config to get expected shape
    data_gen_config = config.get('data_generation', {})
    sim_params = data_gen_config.get('simulation_params', {})
    height, width = sim_params.get('grid_shape', [17, 17]) # Use shape from config
    batch_size = 1 # Test with batch size 1

    # Create dummy initial state and parameters
    initial_state = torch.rand(batch_size, 1, height, width, device=device, dtype=model_dtype)
    # Use dummy scalar params for simplicity, or load spatial if needed/available
    params = {
        'K': torch.tensor(1e-5, device=device, dtype=model_dtype),
        'D': torch.tensor(0.1, device=device, dtype=model_dtype),
        'U': torch.tensor(0.001, device=device, dtype=model_dtype),
        'm': torch.tensor(0.5, device=device, dtype=model_dtype),
        'n': torch.tensor(1.0, device=device, dtype=model_dtype)
    }
    t_target = torch.tensor(sim_params.get('run_time', 500.0), device=device, dtype=model_dtype) # Use run_time from config

    model_input = {'initial_state': initial_state, 'params': params, 't_target': t_target}

    # --- Perform Prediction ---
    try:
        with torch.no_grad():
            # Model might return dict now, handle it
            output = model(model_input, mode='predict_state')
            if isinstance(output, dict):
                prediction = output.get('state')
                if prediction is None:
                     pytest.fail("Model output dictionary missing 'state' key.")
            else:
                # Assume single tensor output is state
                prediction = output
        print(f"Model prediction successful. Output state shape: {prediction.shape}")
    except Exception as e:
        pytest.fail(f"Model prediction failed: {e}")

    # --- Validate Prediction Output ---
    assert isinstance(prediction, torch.Tensor), "Prediction output ('state') is not a tensor."
    expected_shape = (batch_size, model.output_dim, height, width)
    assert prediction.shape == expected_shape, f"Prediction shape mismatch. Expected {expected_shape}, got {prediction.shape}."
    assert prediction.dtype == model_dtype, f"Prediction dtype mismatch. Expected {model_dtype}, got {prediction.dtype}."
    assert not torch.isnan(prediction).any(), "Prediction contains NaNs."
    assert not torch.isinf(prediction).any(), "Prediction contains Infs."

    print("Model prediction test passed!")


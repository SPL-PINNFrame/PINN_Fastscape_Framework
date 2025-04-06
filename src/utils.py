import os
import logging
import random
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf # Import OmegaConf

def setup_logging(log_dir, log_filename="training.log"):
    """Sets up logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_filename)

    # Basic configuration sets up the root logger
    # Avoid reconfiguring if already configured (e.g., by multiple imports)
    # Always configure basic settings and add handlers
    # If basicConfig was called before, it might reset level/format,
    # but adding handlers should still work.
    # A more robust approach might involve getting the root logger and adding handlers manually.
    # For simplicity here, we call basicConfig again, which might reset existing handlers
    # but ensures our desired handlers are present.
    # Consider potential side effects if other parts of the code rely on prior logging config.

    # Get root logger and remove existing handlers to ensure clean setup for this specific call
    # logger = logging.getLogger()
    # for handler in logger.handlers[:]: # Iterate over a copy
    #     logger.removeHandler(handler)

    # Configure logging - This might be too aggressive if called multiple times with different files.
    # A better approach might be to just add the handlers.
    logging.basicConfig(
        level=logging.DEBUG, # Set level to DEBUG for more verbose logging during tests
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler() # Log to console as well
        ],
        force=True # Force reconfiguration if already configured
    )
    logging.info(f"Logging setup complete. Log file: {log_filepath}")
    # else: # Removed the check
    #     # Even if configured, ensure the specific file handler is added?
    #     # This requires more careful handler management.
    #     # For now, basicConfig with force=True should work for the test case.
    #     logging.info("Logger already configured, but ensuring handlers are set.")
    #     # logger = logging.getLogger()
    #     # file_handler_exists = any(isinstance(h, logging.FileHandler) and h.baseFilename == log_filepath for h in logger.handlers)
    #     # if not file_handler_exists:
    #     #     logger.addHandler(logging.FileHandler(log_filepath))
    #     # stream_handler_exists = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    #     # if not stream_handler_exists:
    #     #     logger.addHandler(logging.StreamHandler())



def get_device(config):
    """Gets the appropriate torch device based on config and availability."""
    if config.get('device', 'auto') == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config['device']

    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA specified but not available. Falling back to CPU.")
        device = "cpu"

    logging.info(f"Using device: {device}")
    return torch.device(device)

def set_seed(seed):
    """Sets random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) # for multi-GPU.
            # Ensure deterministic algorithms are used where possible
            # Note: some operations may still be non-deterministic
            # torch.use_deterministic_algorithms(True) # Use if needed, requires PyTorch 1.8+
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logging.info(f"Set random seed to {seed}")

def save_data_sample(data_dict, filepath):
    """Saves a data sample dictionary to a .pt file."""
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(data_dict, filepath)
        # logging.info(f"Successfully saved data sample to {filepath}") # Reduce log verbosity
    except Exception as e:
        logging.error(f"Error saving file {filepath}: {e}")

def load_config(config_path):
    """Loads a YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            # config = yaml.safe_load(f) # Replace yaml.safe_load
            # Use OmegaConf to load and resolve interpolations/calculations
            config = OmegaConf.load(f)
            # Optionally resolve interpolations immediately if needed,
            # though often resolution happens implicitly on access.
            # OmegaConf.resolve(config) # Uncomment if explicit resolution is required
        logging.info(f"Loaded configuration from {config_path} using OmegaConf")
        # Return the OmegaConf object (or convert back to dict if necessary, but OmegaConf object is often preferred)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading config file {config_path}: {e}")
        raise

def save_config(config, filepath):
    """Saves a configuration dictionary to a YAML file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logging.info(f"Configuration saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving config file {filepath}: {e}")


if __name__ == '__main__':
    # Example usage
    # Setup basic logging for standalone run info
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    setup_logging("temp_logs", "utils_test.log")
    logging.info("Testing utils...")

    # Test device selection
    dummy_config = {'device': 'auto'}
    device = get_device(dummy_config)
    logging.info(f"Device selected: {device}")

    # Test seed setting
    set_seed(42)
    logging.info(f"Random float after seed: {random.random()}")

    # Test config loading/saving (create dummy)
    dummy_cfg_path = "temp_logs/dummy_config.yaml"
    dummy_cfg_data = {'learning_rate': 0.001, 'model': {'type': 'MLP'}}
    save_config(dummy_cfg_data, dummy_cfg_path)
    loaded_cfg = load_config(dummy_cfg_path)
    logging.info(f"Loaded config: {loaded_cfg}")

    # Test data saving
    dummy_data = {'tensor': torch.randn(2, 2)}
    save_data_sample(dummy_data, "temp_logs/dummy_sample.pt")

    # Clean up
    import shutil
    if os.path.exists("temp_logs"):
        shutil.rmtree("temp_logs")
        logging.info("Cleaned up temp_logs directory.")

    logging.info("Utils testing done.")

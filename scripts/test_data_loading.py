import argparse
import logging
import os
import sys
import torch # Assuming torch is available in the environment

# Add src directory to Python path to allow importing modules from src
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

try:
    from src.utils import load_config, setup_logging
    from src.data_utils import create_dataloaders
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure the script is run from the project root or the environment has the correct paths.")
    sys.exit(1)

def main(config_path):
    """Tests data loading using create_dataloaders."""
    # --- Setup ---
    config = load_config(config_path)
    log_config = config.get('logging', {})
    # Use a different log file for this test
    log_dir = log_config.get('log_dir', 'logs/component_tests')
    setup_logging(log_dir=log_dir, log_filename="test_data_loading.log")

    logging.info(f"Starting data loading test using config: {config_path}")

    # --- Create DataLoaders ---
    try:
        # create_dataloaders expects the full config dictionary
        # It will internally extract 'data_dir', 'batch_size', 'num_workers', 'dataset_params'
        logging.info("Attempting to create dataloaders...")
        train_loader, val_loader = create_dataloaders(config)
        logging.info("Dataloaders created successfully.")

        # --- Test Loading One Batch ---
        logging.info("Attempting to load one batch from train_loader...")
        first_batch = next(iter(train_loader))

        if first_batch is None:
            logging.error("Failed to load the first batch (collate_fn returned None). Check data files or dataset logic.")
        else:
            logging.info("Successfully loaded one batch from train_loader:")
            for key, value in first_batch.items():
                if isinstance(value, torch.Tensor):
                    logging.info(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    logging.info(f"  - {key}: type={type(value)}, value={value}")
            logging.info("Data loading test successful for one batch.")

    except FileNotFoundError as e:
         logging.error(f"Data loading failed: {e}. Ensure data directory specified in config exists and contains .pt files.")
         logging.error(f"Expected data directory based on config: {config.get('data', {}).get('processed_dir', 'data/processed')}")
    except Exception as e:
        logging.error("An unexpected error occurred during data loading test:", exc_info=True)

if __name__ == "__main__":
    # Use train_config.yaml by default for testing data loading part
    default_config = os.path.join(project_root, 'configs', 'train_config.yaml')

    parser = argparse.ArgumentParser(description="Test data loading functionality.")
    parser.add_argument('--config', type=str, default=default_config,
                        help=f'Path to the configuration file (default: {default_config})')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)

    main(args.config)
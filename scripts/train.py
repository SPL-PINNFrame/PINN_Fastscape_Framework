import argparse
import logging
import os
import sys
import time
import torch
# Add src directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.utils import load_config, setup_logging, set_seed
from src.data_utils import create_dataloaders # Assuming data_utils.py has this
from src.models import FastscapePINN # Use the new enhanced model
from src.trainer import PINNTrainer # Assuming trainer.py has this

def main(args):
    """Main function to train the PINN model."""
    config = load_config(args.config)
    train_config = config.get('training', {})
    # Define output and log directories early for logging setup
    output_dir = config.get('output_dir', 'results/')
    # Get run_name from config, PINNTrainer will handle default if not provided
    run_name = config.get('run_name', train_config.get('run_name'))
    log_dir = os.path.join(output_dir, run_name, 'logs')
    os.makedirs(log_dir, exist_ok=True) # Ensure log directory exists
    # Combine log_dir and a default filename for the log_file argument
    log_file_path = os.path.join(log_dir, 'train.log')
    # Get log level from config, default to INFO
    log_level = config.get('logging', {}).get('log_level', 'INFO')
    setup_logging(log_level=log_level, log_file=log_file_path, log_to_console=True) # Pass the full path

    # Set random seed for reproducibility
    seed = train_config.get('seed', 42)
    set_seed(seed)
    logging.info(f"Random seed set to {seed}")

    # Create DataLoaders
    data_config = config.get('data', {})
    processed_data_dir = data_config.get('processed_dir', 'data/processed')
    batch_size = train_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 0)
    try:
        # Pass data_config, it might contain necessary info beyond the explicit args
        # Assuming the first positional argument is config, based on previous error.
        # Explicitly pass other known arguments as keywords. Needs verification.
        # Assuming create_dataloaders takes the config dictionary as its primary argument
        # and extracts needed values internally.
        # Assuming create_dataloaders returns only train and val loaders
        # Pass the full config, as create_dataloaders needs 'training' section for batch_size
        # and 'data' section for processed_dir, splits, etc.
        dataloaders = create_dataloaders(config) # Get the dictionary
        train_loader = dataloaders.get('train')
        val_loader = dataloaders.get('val')
        test_loader = dataloaders.get('test') # Get the test loader as well
    except ValueError as e:
         logging.error(f"Failed to create dataloaders: {e}")
         sys.exit(1)


    # Initialize Model
    model_config = config.get('model', {}).copy() # Get a copy to modify
    model_name = model_config.pop('name', 'FastscapePINN') # Get name and remove it, default if missing
    model_dtype_str = model_config.pop('dtype', 'float32') # Get dtype and remove it
    model_dtype = torch.float32 if model_dtype_str == 'float32' else torch.float64

    # Select model class based on name
    if model_name == 'AdaptiveFastscapePINN':
        from src.models import AdaptiveFastscapePINN as ModelClass
    elif model_name == 'FastscapePINN':
        from src.models import FastscapePINN as ModelClass
    elif model_name == 'MLP_PINN':
        from src.models import MLP_PINN as ModelClass
    else:
        logging.error(f"Unknown model name '{model_name}' specified in config.")
        sys.exit(1)

    # Instantiate the selected model with remaining config args and dtype
    # Pass only the relevant args from model_config to the constructor
    # Inspecting model constructors: they take specific args, not **kwargs generally
    # We need to filter model_config based on ModelClass.__init__ signature
    # For simplicity now, assume model_config only contains valid args after popping 'name', 'dtype'

    # --- Explicitly add domain_x and domain_y if the model expects them ---
    # This ensures that interpolated values from the main config are passed correctly.
    if model_name == 'AdaptiveFastscapePINN':
        # Check if they exist in the main config (they should due to interpolation)
        if 'physics_params' in config and 'domain_x' in config['physics_params']:
            model_config['domain_x'] = config['physics_params']['domain_x']
        else:
            logging.warning("Could not find resolved 'physics_params.domain_x' in config for AdaptiveFastscapePINN.")
        if 'physics_params' in config and 'domain_y' in config['physics_params']:
            model_config['domain_y'] = config['physics_params']['domain_y']
        else:
            logging.warning("Could not find resolved 'physics_params.domain_y' in config for AdaptiveFastscapePINN.")
        # Remove potentially unresolved keys if they exist from the copy
        model_config.pop('domain_x', None) # Remove if it was copied as a string "${...}"
        model_config.pop('domain_y', None) # Remove if it was copied as a string "${...}"


    # TODO: Implement proper argument filtering based on signature if needed
    try:
        model = ModelClass(**model_config).to(dtype=model_dtype)
    except TypeError as e:
        logging.error(f"Error instantiating model '{model_name}' with config {model_config}: {e}")
        # Log expected arguments vs provided arguments if possible
        import inspect
        sig = inspect.signature(ModelClass.__init__)
        logging.error(f"Expected arguments for {model_name}.__init__: {sig}")
        sys.exit(1)
    logging.info("Model initialized.")
    # Log model structure or parameter count if desired
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model has {num_params:,} trainable parameters.")


    # Initialize Trainer
    # Pass the full config to the trainer as it might need various sections
    trainer = PINNTrainer(model, config, train_loader, val_loader)
    logging.info("Trainer initialized.")

    # Start Training
    logging.info("Starting training process...")
    trainer.train()
    logging.info("Training process finished.")

    # Optional: Evaluate on test set after training
    run_test = train_config.get('run_test_evaluation', False)
    if run_test and test_loader is not None:
        logging.info("Running evaluation on the test set...")
        # Need to implement an evaluate method in the Trainer or a separate script
        # test_loss, test_metrics = trainer.evaluate(test_loader)
        # logging.info(f"Test Loss: {test_loss:.4f}, Test Metrics: {test_metrics}")
        logging.warning("Test set evaluation not implemented yet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Fastscape PINN model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    main(args)

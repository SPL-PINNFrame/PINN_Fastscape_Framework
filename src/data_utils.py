import os
import glob
import logging
import json
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import random

# Placeholder for actual data loading format
# Assume data is saved as .pt files, each containing a dictionary:
# {'initial_topo': tensor, 'final_topo': tensor, 'uplift_rate': tensor/float/array, ...}

# (TerrainDataNormalizer class removed)


class FastscapeDataset(Dataset):
    """PyTorch Dataset for loading Fastscape simulation data."""
    def __init__(self, file_list, normalize=False, norm_stats=None, transform=None):
        """
        Args:
            file_list (list): List of paths to the data files for this dataset split.
            normalize (bool): Whether to apply normalization. Defaults to False.
            norm_stats (dict, optional): Dictionary containing min/max statistics for normalization.
                                         Required if normalize is True. Defaults to None.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_list = file_list
        self.normalize = normalize
        self.norm_stats = norm_stats
        self.transform = transform
        self.epsilon = 1e-8 # Small value to prevent division by zero in normalization

        if not self.file_list:
            logging.warning(f"Received an empty file list for this dataset split.")

        if self.normalize and self.norm_stats is None:
            logging.warning("Normalization enabled but no norm_stats provided. Data will not be normalized.")
            self.normalize = False # Disable normalization if stats are missing

        if self.normalize:
            logging.info(f"FastscapeDataset initialized with Min-Max normalization enabled.")
            # Log available stats keys for debugging
            logging.debug(f"Normalization stats available for keys: {list(self.norm_stats.keys())}")
        else:
            logging.info("FastscapeDataset initialized without normalization.")


    def __len__(self):
        """Returns the total number of samples."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index.
        Handles both scalar and spatial (numpy array) parameters saved in .pt files.
        Applies normalization using the provided normalizer.
        """
        filepath = self.file_list[idx]
        try:
            # Load data from file
            # Note: weights_only=False is used because .pt files may contain numpy arrays,
            # which are disallowed by default with weights_only=True for security reasons.
            # Ensure the data source is trusted.
            sample_data = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)

            # --- Extract required fields ---
            initial_topo = sample_data.get('initial_topo')
            final_topo = sample_data.get('final_topo')
            uplift_rate = sample_data.get('uplift_rate')
            k_f = sample_data.get('k_f')
            k_d = sample_data.get('k_d')
            m = sample_data.get('m')
            n = sample_data.get('n')
            run_time = sample_data.get('run_time')

            # Check if all required fields are loaded
            required_keys = ['initial_topo', 'final_topo', 'uplift_rate', 'k_f', 'k_d', 'm', 'n', 'run_time']
            missing_keys = [key for key in required_keys if sample_data.get(key) is None]
            if missing_keys:
                 raise ValueError(f"Missing required data fields {missing_keys} in {filepath}")

            # --- Preprocessing ---
            # 1. Ensure correct tensor types (e.g., float32)
            # Function to safely convert scalar/numpy/tensor to FloatTensor
            def to_float_tensor(value):
                if isinstance(value, torch.Tensor):
                    return value.float()
                elif isinstance(value, np.ndarray):
                    return torch.tensor(value, dtype=torch.float32)
                elif isinstance(value, (int, float)):
                    return torch.tensor(float(value), dtype=torch.float32) # Ensure float conversion
                else:
                    raise TypeError(f"Unsupported type for parameter conversion: {type(value)}")

            initial_topo = to_float_tensor(initial_topo)
            final_topo = to_float_tensor(final_topo)
            uplift_rate = to_float_tensor(uplift_rate)
            k_f = to_float_tensor(k_f)
            k_d = to_float_tensor(k_d)
            m = to_float_tensor(m)
            n = to_float_tensor(n)
            run_time = to_float_tensor(run_time)

            # 2. Min-Max Normalization
            if self.normalize:
                # Define which fields to normalize and their corresponding keys in norm_stats
                # It's often useful to normalize initial and final topo using the same stats
                fields_to_normalize = {
                    'initial_topo': 'topo', # Use combined 'topo' stats
                    'final_topo': 'topo',
                    'uplift_rate': 'uplift_rate',
                    'k_f': 'k_f',
                    'k_d': 'k_d',
                    # m, n, run_time are usually not normalized
                }

                def _normalize_field(tensor, field_key):
                    stats = self.norm_stats.get(field_key)
                    if stats and 'min' in stats and 'max' in stats:
                        min_val = torch.tensor(stats['min'], device=tensor.device, dtype=tensor.dtype)
                        max_val = torch.tensor(stats['max'], device=tensor.device, dtype=tensor.dtype)
                        range_val = max_val - min_val
                        # Apply normalization: (value - min) / (range + epsilon)
                        return (tensor - min_val) / (range_val + self.epsilon)
                    else:
                        # logging.warning(f"Normalization stats missing for key '{field_key}'. Skipping normalization for this field.")
                        return tensor

                initial_topo = _normalize_field(initial_topo, fields_to_normalize['initial_topo'])
                final_topo = _normalize_field(final_topo, fields_to_normalize['final_topo'])
                uplift_rate = _normalize_field(uplift_rate, fields_to_normalize['uplift_rate'])
                k_f = _normalize_field(k_f, fields_to_normalize['k_f'])
                k_d = _normalize_field(k_d, fields_to_normalize['k_d'])

            # 3. Apply other transforms if provided
            if self.transform:
                 # Example: sample = self.transform(sample)
                 pass # Placeholder

            # --- Prepare output dictionary ---
            output = {
                'initial_topo': initial_topo,
                'uplift_rate': uplift_rate,
                'final_topo': final_topo,
                'k_f': k_f,
                'k_d': k_d,
                'm': m,
                'n': n,
                'run_time': run_time,
                'target_shape': final_topo.shape # Keep target_shape if needed
            }

            return output

        except Exception as e:
            logging.error(f"Error loading or processing sample {idx} from {filepath}: {e}", exc_info=True) # Log traceback
            return None # Return None on error

    def denormalize_state(self, normalized_state_tensor):
        """
        Denormalizes a state tensor (e.g., predicted topography) using stored Min-Max stats.
        Assumes the state tensor corresponds to the 'topo' field statistics.

        Args:
            normalized_state_tensor (torch.Tensor): The normalized state tensor.

        Returns:
            torch.Tensor: The denormalized state tensor.
        """
        if not self.normalize or self.norm_stats is None:
            # logging.warning("Denormalization called but normalization was not enabled or stats are missing.")
            return normalized_state_tensor # Return as is if no normalization applied

        topo_stats = self.norm_stats.get('topo')
        if topo_stats and 'min' in topo_stats and 'max' in topo_stats:
            min_val = torch.tensor(topo_stats['min'], device=normalized_state_tensor.device, dtype=normalized_state_tensor.dtype)
            max_val = torch.tensor(topo_stats['max'], device=normalized_state_tensor.device, dtype=normalized_state_tensor.dtype)
            range_val = max_val - min_val
            # Apply denormalization: normalized * (range + epsilon) + min
            return normalized_state_tensor * (range_val + self.epsilon) + min_val
        else:
            logging.warning("Normalization stats for 'topo' missing. Cannot denormalize state.")
            return normalized_state_tensor # Return as is if stats are missing

# --- Utility function to create DataLoaders ---

def create_dataloaders(config):
    """Creates train, validation, and test dataloaders with normalization handling."""

    # --- Define collate_fn first ---
    def collate_fn_filter_none(batch):
        """Custom collate_fn that filters out None results from __getitem__."""
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
             return None # Return None if the whole batch failed
        try:
             # Use default collate to combine samples into a batch
             return torch.utils.data.dataloader.default_collate(batch)
        except Exception as e:
             logging.error(f"Error during default_collate: {e}. Batch content structure might be inconsistent.")
             # Log details about the batch structure if possible
             if batch:
                  logging.error(f"First item keys: {batch[0].keys() if isinstance(batch[0], dict) else 'Not a dict'}")
             return None
    """Creates train, validation, and test dataloaders with normalization handling."""
    data_config = config.get('data', {})
    norm_config = data_config.get('normalization', {})
    train_config = config.get('training', {})

    data_dir = data_config.get('processed_dir', 'data/processed') # Default relative to project root
    batch_size = train_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 0)
    train_split = data_config.get('train_split', 0.8)
    val_split = data_config.get('val_split', 0.1)
    test_split = 1.0 - train_split - val_split # Calculate test split

    logging.info(f"Creating dataloaders from: {data_dir}")
    logging.info(f"Batch size: {batch_size}, Num workers: {num_workers}")
    logging.info(f"Train split: {train_split}, Validation split: {val_split}")

    # --- Find data files in all resolution subdirectories ---
    all_files = []
    if os.path.isdir(data_dir):
        # Check for resolution subdirectories first
        resolution_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('resolution_')]
        if resolution_dirs:
            logging.info(f"Found resolution subdirectories: {resolution_dirs}. Searching for .pt files within them.")
            for res_dir in resolution_dirs:
                # Use recursive glob to find .pt files in each resolution subdir
                files_in_res = glob.glob(os.path.join(data_dir, res_dir, '**', '*.pt'), recursive=True)
                all_files.extend(files_in_res)
        else:
            # If no resolution subdirs, search directly in data_dir (backward compatibility?)
            logging.info(f"No resolution subdirectories found in {data_dir}. Searching for .pt files directly.")
            all_files = glob.glob(os.path.join(data_dir, '**', '*.pt'), recursive=True)
    else:
         logging.error(f"Data directory not found or is not a directory: {data_dir}")


    if not all_files:
        logging.error(f"No .pt files found in {data_dir} or its resolution subdirectories. Cannot create dataloaders.")
        raise FileNotFoundError(f"No .pt files found in the specified data directory structure: {data_dir}")

    # Shuffle files before splitting
    random.shuffle(all_files)
    num_total = len(all_files)
    num_train = int(num_total * train_split)
    num_val = int(num_total * val_split)

    # Ensure validation set gets at least one sample if split > 0 and possible
    if val_split > 0 and num_val == 0 and num_total > num_train:
        num_val = 1
    # Adjust train count if validation took the last sample(s)
    num_train = min(num_train, num_total - num_val)
    num_test = num_total - num_train - num_val

    train_files = all_files[:num_train]
    val_files = all_files[num_train : num_train + num_val]
    test_files = all_files[num_train + num_val:] # Files remaining for test set

    logging.info(f"Total files found: {num_total}")
    logging.info(f"Splitting into: Train={len(train_files)}, Validation={len(val_files)}")
    logging.info(f"Test files count: {len(test_files)}")

    # --- Normalization Handling (Min-Max) ---
    normalize_data = norm_config.get('enabled', False)
    norm_stats = None
    if normalize_data:
        stats_file = norm_config.get('stats_file', None) # Path relative to output_dir usually
        compute_stats = norm_config.get('compute_stats', False)
        # Define fields to compute stats for (match fields_to_normalize in Dataset)
        fields_for_stats = ['topo', 'uplift_rate', 'k_f', 'k_d'] # Corresponds to keys used in norm_stats

        stats_loaded_or_computed = False
        if stats_file and os.path.exists(stats_file):
            logging.info(f"Attempting to load normalization stats from: {stats_file}")
            try:
                with open(stats_file, 'r') as f:
                    norm_stats = json.load(f)
                logging.info(f"Normalization stats loaded successfully from {stats_file}.")
                stats_loaded_or_computed = True
            except Exception as e:
                logging.error(f"Failed to load normalization stats from {stats_file}: {e}. Will attempt to compute if enabled.")

        if not stats_loaded_or_computed and compute_stats:
            if not train_files:
                logging.warning("Cannot compute normalization stats: No training files available.")
            else:
                logging.info("Computing Min-Max normalization statistics from the training set...")
                norm_stats = {}
                # Initialize min/max trackers
                for field in fields_for_stats:
                    norm_stats[field] = {'min': float('inf'), 'max': float('-inf')}

                # Iterate through training files to compute stats
                num_processed = 0
                for f_path in train_files:
                    try:
                        # Note: weights_only=False needed for potential numpy arrays in data.
                        data = torch.load(f_path, map_location='cpu', weights_only=False)
                        num_processed += 1

                        # Helper to update stats for a tensor
                        def _update_stats(tensor_val, field_key):
                            if isinstance(tensor_val, torch.Tensor):
                                current_min = tensor_val.min().item()
                                current_max = tensor_val.max().item()
                                norm_stats[field_key]['min'] = min(norm_stats[field_key]['min'], current_min)
                                norm_stats[field_key]['max'] = max(norm_stats[field_key]['max'], current_max)
                            elif isinstance(tensor_val, (int, float)): # Handle scalar tensors/numbers
                                norm_stats[field_key]['min'] = min(norm_stats[field_key]['min'], float(tensor_val))
                                norm_stats[field_key]['max'] = max(norm_stats[field_key]['max'], float(tensor_val))
                            elif isinstance(tensor_val, np.ndarray): # Handle numpy arrays
                                current_min = float(tensor_val.min())
                                current_max = float(tensor_val.max())
                                norm_stats[field_key]['min'] = min(norm_stats[field_key]['min'], current_min)
                                norm_stats[field_key]['max'] = max(norm_stats[field_key]['max'], current_max)


                        # Update stats for relevant fields
                        # Use 'topo' key for both initial and final topo stats
                        if 'initial_topo' in data: _update_stats(data['initial_topo'], 'topo')
                        if 'final_topo' in data: _update_stats(data['final_topo'], 'topo')
                        if 'uplift_rate' in data: _update_stats(data['uplift_rate'], 'uplift_rate')
                        if 'k_f' in data: _update_stats(data['k_f'], 'k_f')
                        if 'k_d' in data: _update_stats(data['k_d'], 'k_d')

                    except Exception as e:
                        logging.warning(f"Skipping file {f_path} during stats computation due to error: {e}")

                if num_processed > 0:
                    logging.info(f"Min-Max stats computed from {num_processed} training files.")
                    logging.debug(f"Computed norm_stats: {norm_stats}")
                    stats_loaded_or_computed = True
                    if stats_file:
                        try:
                            os.makedirs(os.path.dirname(stats_file), exist_ok=True)
                            with open(stats_file, 'w') as f:
                                json.dump(norm_stats, f, indent=2)
                            logging.info(f"Normalization statistics saved to: {stats_file}")
                        except Exception as e:
                            logging.error(f"Failed to save normalization stats to {stats_file}: {e}")
                else:
                    logging.error("Failed to compute normalization stats: No files processed successfully.")


        if not stats_loaded_or_computed:
            logging.warning("Normalization enabled, but no stats were loaded or computed. Disabling normalization.")
            normalize_data = False
            norm_stats = None

    else:
        logging.info("Normalization is disabled in the configuration.")

    # --- Create Datasets and DataLoaders ---
    # --- Create Datasets and DataLoaders ---
    # Pass normalize flag and computed/loaded norm_stats to the datasets
    train_dataset = FastscapeDataset(train_files, normalize=normalize_data, norm_stats=norm_stats)
    val_dataset = FastscapeDataset(val_files, normalize=normalize_data, norm_stats=norm_stats)
    test_dataset = FastscapeDataset(test_files, normalize=normalize_data, norm_stats=norm_stats)


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_filter_none, # Use the defined function
        persistent_workers=num_workers > 0 # Add persistent workers if num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_filter_none,
        persistent_workers=num_workers > 0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_filter_none,
        persistent_workers=num_workers > 0
    )

    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")

    # Return only train and val loaders for now, consistent with train.py usage
    # MODIFIED: Return all three loaders
    return {'train': train_loader, 'val': val_loader, 'test': test_loader} # Return a dictionary


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Use INFO for cleaner output
    print("Testing data utilities with TerrainDataNormalizer...")

    dummy_data_root = 'data_dummy/processed'
    dummy_stats_file = 'data_dummy/norm_stats.json'
    os.makedirs(dummy_data_root, exist_ok=True)
    if os.path.exists(dummy_stats_file):
        os.remove(dummy_stats_file)

    # Create dummy files
    for i in range(10): # Create more files for better stats
        is_spatial = i % 2 == 0
        dummy_sample = {
            'initial_topo': torch.rand(1, 64, 64) * 100 + i, # Add variance
            'final_topo': torch.rand(1, 64, 64) * 100 + i + 5,
            'uplift_rate': np.random.rand(64, 64).astype(np.float32) * 0.01 + i*0.001 if is_spatial else torch.rand(1).item() * 0.01 + i*0.001,
            'k_f': np.random.rand(64, 64).astype(np.float32) * 1e-5 + i*1e-6 if is_spatial else torch.rand(1).item() * 1e-5 + i*1e-6,
            'k_d': np.random.rand(64, 64).astype(np.float32) * 0.1 + i*0.01 if is_spatial else torch.rand(1).item() * 0.1 + i*0.01,
            'm': 0.5,
            'n': 1.0,
            'run_time': 5000.0 + i * 100
        }
        torch.save(dummy_sample, os.path.join(dummy_data_root, f'sample_{i}.pt'))

    # --- Test Case 1: Compute Stats ---
    print("\n--- Test Case 1: Compute Stats ---")
    dummy_config_compute = {
        'data': {
            'processed_dir': dummy_data_root,
            'train_split': 0.7,
            'val_split': 0.3,
            'num_workers': 0,
            'normalization': {
                'enabled': True,
                'mode': 'standardize',
                'compute_stats': True,
                'stats_file': dummy_stats_file,
                'fields': ['initial_topo', 'final_topo', 'uplift_rate', 'k_f', 'k_d'] # Specify fields
            }
        },
        'training': {'batch_size': 4}
    }

    try:
        train_loader_c, val_loader_c = create_dataloaders(dummy_config_compute)
        print("Dataloaders created (compute stats).")
        print(f"Stats file created: {os.path.exists(dummy_stats_file)}")

        # Check a batch from train loader
        print("Checking first batch from train_loader (should be normalized):")
        first_batch_c = next(iter(train_loader_c))
        if first_batch_c:
            print("Sample initial_topo mean (normalized):", first_batch_c['initial_topo'].mean().item())
            print("Sample uplift_rate mean (normalized):", first_batch_c['uplift_rate'].mean().item())
        else:
            print("Failed to get first batch.")

    except Exception as e:
        print(f"\nError during compute stats test: {e}")
        import traceback
        traceback.print_exc()

    # --- Test Case 2: Load Stats ---
    print("\n--- Test Case 2: Load Stats ---")
    dummy_config_load = {
        'data': {
            'processed_dir': dummy_data_root,
            'train_split': 0.7,
            'val_split': 0.3,
            'num_workers': 0,
            'normalization': {
                'enabled': True,
                'mode': 'standardize', # Mode here is less important if loading
                'compute_stats': False, # Don't recompute
                'stats_file': dummy_stats_file
            }
        },
        'training': {'batch_size': 4}
    }
    try:
        train_loader_l, val_loader_l = create_dataloaders(dummy_config_load)
        print("Dataloaders created (load stats).")

        # Check a batch from val loader
        print("Checking first batch from val_loader (should be normalized using loaded stats):")
        first_batch_l = next(iter(val_loader_l))
        if first_batch_l:
             print("Sample final_topo mean (normalized):", first_batch_l['final_topo'].mean().item())
             print("Sample k_d mean (normalized):", first_batch_l['k_d'].mean().item())
             # Example denormalization
             if hasattr(val_loader_l.dataset, 'normalizer') and val_loader_l.dataset.normalizer:
                  denorm_topo = val_loader_l.dataset.normalizer.denormalize(first_batch_l['final_topo'][0], 'topo')
                  print("Sample denormalized final_topo mean:", denorm_topo.mean().item())
             else:
                  print("Normalizer not found on dataset for denormalization test.")

        else:
             print("Failed to get first batch.")

    except Exception as e:
        print(f"\nError during load stats test: {e}")
        import traceback
        traceback.print_exc()

    # --- Test Case 3: Normalization Disabled ---
    print("\n--- Test Case 3: Normalization Disabled ---")
    dummy_config_disabled = {
        'data': {
            'processed_dir': dummy_data_root,
            'train_split': 0.7,
            'val_split': 0.3,
            'num_workers': 0,
            'normalization': {'enabled': False} # Disabled
        },
        'training': {'batch_size': 4}
    }
    try:
        train_loader_d, val_loader_d = create_dataloaders(dummy_config_disabled)
        print("Dataloaders created (normalization disabled).")
        print("Checking first batch from train_loader (should NOT be normalized):")
        first_batch_d = next(iter(train_loader_d))
        if first_batch_d:
            print("Sample initial_topo mean (original):", first_batch_d['initial_topo'].mean().item())
        else:
            print("Failed to get first batch.")

    except Exception as e:
        print(f"\nError during disabled normalization test: {e}")
        import traceback
        traceback.print_exc()


    finally:
        import shutil
        if os.path.exists('data_dummy'):
            print("\nCleaning up dummy data...")
            shutil.rmtree('data_dummy')

    print("\nData utilities testing done.")

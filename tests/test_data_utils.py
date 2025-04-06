import pytest
import torch
import numpy as np
# import netCDF4 # No longer needed for .pt files
import torch.testing as tt # For numerical comparisons
import os
import tempfile
from omegaconf import OmegaConf

# Adjust import path if necessary
from src.data_utils import FastscapeDataset, create_dataloaders

# --- Fixture for creating temporary NetCDF test data ---

@pytest.fixture(scope="module")
def temp_pt_data():
    """Creates temporary .pt files for testing FastscapeDataset."""
    tmpdir = tempfile.TemporaryDirectory()
    data_path = tmpdir.name
    num_samples = 3
    rows, cols = 10, 10
    # Define keys consistent with FastscapeDataset expectations
    state_keys = ['initial_topo', 'final_topo']
    param_keys_scalar = ['m', 'n', 'run_time'] # Fixed exponents, time
    param_keys_spatial_or_scalar = ['uplift_rate', 'k_f', 'k_d'] # Can be scalar or spatial

    file_paths = []
    for i in range(num_samples):
        filepath = os.path.join(data_path, f"sample_{i:05d}.pt")
        sample_data = {}

        # State variables
        # initial_topo: ramp + noise, range approx [0, 3]
        sample_data['initial_topo'] = torch.rand(1, rows, cols) + \
                                      torch.linspace(0, 1, rows).unsqueeze(1) + \
                                      torch.linspace(0, 1, cols).unsqueeze(0)
        # final_topo: similar but slightly shifted
        sample_data['final_topo'] = torch.rand(1, rows, cols) + \
                                    torch.linspace(0.1, 1.1, rows).unsqueeze(1) + \
                                    torch.linspace(0.1, 1.1, cols).unsqueeze(0)

        # Scalar parameters (fixed values for simplicity in testing normalization)
        sample_data['m'] = torch.tensor(0.5)
        sample_data['n'] = torch.tensor(1.0)
        sample_data['run_time'] = torch.tensor(1000.0 + i * 100) # Varies slightly

        # Spatial or Scalar parameters (let's make some vary)
        # uplift_rate: spatial, range approx [0, 0.01 * 3] = [0, 0.03]
        sample_data['uplift_rate'] = torch.rand(rows, cols) * 0.01 * (i + 1)
        # k_f: scalar, range [1e-5, 3e-5]
        sample_data['k_f'] = torch.tensor(1e-5 * (i + 1))
        # k_d: spatial, range approx [0, 0.1 * 3] = [0, 0.3]
        sample_data['k_d'] = torch.rand(rows, cols) * 0.1 * (i + 1)

        # Save as .pt file
        torch.save(sample_data, filepath)

        file_paths.append(filepath)

    # Yield the directory path and cleanup afterwards
    # Define the keys used for normalization stats calculation later
    norm_keys = ['topo', 'uplift_rate', 'k_f', 'k_d']
    yield data_path, file_paths, norm_keys, rows, cols

    # Cleanup
    tmpdir.cleanup()

# --- Tests for FastscapeDataset ---

def test_dataset_initialization_basic(temp_pt_data):
    """Tests basic initialization of FastscapeDataset without normalization."""
    data_path, file_paths, _, _, _ = temp_pt_data
    # No need for config here, just pass file list
    dataset = FastscapeDataset(file_paths, normalize=False)
    assert dataset is not None
    assert len(dataset.file_list) == len(file_paths)
    assert not dataset.normalize
    assert dataset.norm_stats is None

def test_dataset_len(temp_pt_data):
    """Tests the __len__ method."""
    _, file_paths, _, _, _ = temp_pt_data
    num_samples = len(file_paths)
    dataset = FastscapeDataset(file_paths)
    assert len(dataset) == num_samples


def test_dataset_getitem_structure(temp_pt_data):
    """Tests the structure and basic types of items returned by __getitem__ (no normalization)."""
    _, file_paths, _, rows, cols = temp_pt_data
    dataset = FastscapeDataset(file_paths, normalize=False)
    sample = dataset[0] # Get the first sample

    assert isinstance(sample, dict)
    # Check expected keys based on the new structure
    expected_keys = ['initial_topo', 'final_topo', 'uplift_rate', 'k_f', 'k_d', 'm', 'n', 'run_time', 'target_shape']
    for key in expected_keys:
        assert key in sample, f"Key '{key}' missing from dataset sample"

    # Check shapes and types
    assert isinstance(sample['initial_topo'], torch.Tensor)
    assert sample['initial_topo'].shape == (1, rows, cols) # Expect (C, H, W)
    assert sample['initial_topo'].dtype == torch.float32

    assert isinstance(sample['final_topo'], torch.Tensor)
    assert sample['final_topo'].shape == (1, rows, cols)
    assert sample['final_topo'].dtype == torch.float32

    assert isinstance(sample['uplift_rate'], torch.Tensor)
    # Shape depends if it was saved as scalar or spatial in the fixture
    # The fixture saves it as spatial (rows, cols), dataset converts to (rows, cols) tensor
    assert sample['uplift_rate'].shape == (rows, cols)
    assert sample['uplift_rate'].dtype == torch.float32

    assert isinstance(sample['k_f'], torch.Tensor)
    assert sample['k_f'].ndim == 0 # Scalar tensor
    assert sample['k_f'].dtype == torch.float32

    assert isinstance(sample['k_d'], torch.Tensor)
    assert sample['k_d'].shape == (rows, cols)
    assert sample['k_d'].dtype == torch.float32

    assert isinstance(sample['m'], torch.Tensor)
    assert sample['m'].ndim == 0
    assert sample['m'].dtype == torch.float32

    assert isinstance(sample['n'], torch.Tensor)
    assert sample['n'].ndim == 0
    assert sample['n'].dtype == torch.float32

    assert isinstance(sample['run_time'], torch.Tensor)
    assert sample['run_time'].ndim == 0
    assert sample['run_time'].dtype == torch.float32

    assert sample['target_shape'] == (1, rows, cols)


def test_dataset_getitem_values(temp_pt_data):
    """Tests the actual values loaded by __getitem__ against the source .pt file (no normalization)."""
    _, file_paths, _, _, _ = temp_pt_data
    dataset = FastscapeDataset(file_paths, normalize=False)

    # Check values for the first sample (index 0)
    sample_idx = 0
    sample = dataset[sample_idx]
    source_filepath = file_paths[sample_idx]

    # Load the original data directly from the .pt file
    original_data = torch.load(source_filepath, map_location='cpu')

    # Compare loaded sample values with original data
    # Need to handle potential type/shape differences introduced by to_float_tensor
    tt.assert_close(sample['initial_topo'], original_data['initial_topo'].float())
    tt.assert_close(sample['final_topo'], original_data['final_topo'].float())

    # Handle uplift_rate (might be numpy in original)
    if isinstance(original_data['uplift_rate'], np.ndarray):
        tt.assert_close(sample['uplift_rate'], torch.from_numpy(original_data['uplift_rate']).float())
    else:
        tt.assert_close(sample['uplift_rate'], original_data['uplift_rate'].float())

    tt.assert_close(sample['k_f'], original_data['k_f'].float())

    if isinstance(original_data['k_d'], np.ndarray):
        tt.assert_close(sample['k_d'], torch.from_numpy(original_data['k_d']).float())
    else:
        tt.assert_close(sample['k_d'], original_data['k_d'].float())

    tt.assert_close(sample['m'], original_data['m'].float())
    tt.assert_close(sample['n'], original_data['n'].float())
    tt.assert_close(sample['run_time'], original_data['run_time'].float())


# @pytest.mark.skip(reason="Normalization test needs implementation based on actual normalization logic")
def test_dataset_normalization(temp_pt_data):
    """Tests the Min-Max normalization logic if normalize=True."""
    _, file_paths, norm_keys, _, _ = temp_pt_data

    # --- Calculate Approximate Expected Stats from Fixture Logic ---
    # These are rough estimates based on how temp_pt_data generates values
    expected_stats = {
        'topo': {'min': 0.0, 'max': 3.2}, # ramp [0,2] + noise [0,1] + shift [0.1, 1.1] -> max ~ 1+1+1.1+noise ~ 3.2
        'uplift_rate': {'min': 0.0, 'max': 0.03}, # rand[0,1]*0.01*(i+1), max i=2 -> max ~ 0.03
        'k_f': {'min': 1e-5, 'max': 3e-5}, # 1e-5 * (i+1)
        'k_d': {'min': 0.0, 'max': 0.3} # rand[0,1]*0.1*(i+1) -> max ~ 0.3
        # m, n, run_time are not normalized by default in the current implementation
    }
    epsilon = 1e-8 # Epsilon used in dataset normalization

    # --- Create Dataset with Normalization ---
    dataset = FastscapeDataset(file_paths, normalize=True, norm_stats=expected_stats)
    assert dataset.normalize
    assert dataset.norm_stats == expected_stats

    # --- Get a Sample and Check Normalization ---
    sample_idx = 1 # Check a sample in the middle
    sample = dataset[sample_idx]
    original_data = torch.load(file_paths[sample_idx], map_location='cpu')

    # Check normalized state variables (topo)
    topo_stats = expected_stats['topo']
    topo_range = topo_stats['max'] - topo_stats['min']
    expected_norm_init_topo = (original_data['initial_topo'].float() - topo_stats['min']) / (topo_range + epsilon)
    expected_norm_final_topo = (original_data['final_topo'].float() - topo_stats['min']) / (topo_range + epsilon)

    tt.assert_close(sample['initial_topo'], expected_norm_init_topo, atol=1e-6, rtol=1e-5)
    tt.assert_close(sample['final_topo'], expected_norm_final_topo, atol=1e-6, rtol=1e-5)
    # Check bounds [0, 1] (allow small tolerance)
    assert sample['initial_topo'].min() >= -1e-6
    assert sample['initial_topo'].max() <= 1.0 + 1e-6
    assert sample['final_topo'].min() >= -1e-6
    assert sample['final_topo'].max() <= 1.0 + 1e-6

    # Check normalized parameters
    def check_param_norm(param_key):
        stats = expected_stats[param_key]
        param_range = stats['max'] - stats['min']
        original_val = original_data[param_key]
        if isinstance(original_val, np.ndarray):
            original_val = torch.from_numpy(original_val).float()
        else:
            original_val = original_val.float()
        expected_norm_val = (original_val - stats['min']) / (param_range + epsilon)
        tt.assert_close(sample[param_key], expected_norm_val, atol=1e-6, rtol=1e-5)
        assert sample[param_key].min() >= -1e-6
        assert sample[param_key].max() <= 1.0 + 1e-6

    check_param_norm('uplift_rate')
    check_param_norm('k_f')
    check_param_norm('k_d')

    # Check non-normalized params are unchanged
    tt.assert_close(sample['m'], original_data['m'].float())
    tt.assert_close(sample['n'], original_data['n'].float())
    tt.assert_close(sample['run_time'], original_data['run_time'].float())

    # --- Test Denormalization ---
    denormalized_final_topo = dataset.denormalize_state(sample['final_topo'])
    original_final_topo = original_data['final_topo'].float()
    # Denormalized should be close to original
    tt.assert_close(denormalized_final_topo, original_final_topo, atol=1e-5, rtol=1e-4)

    # Test denormalization when normalization is off
    dataset_no_norm = FastscapeDataset(file_paths, normalize=False)
    sample_no_norm = dataset_no_norm[sample_idx]
    denorm_state_no_norm = dataset_no_norm.denormalize_state(sample_no_norm['final_topo'])
    # Should return the tensor unchanged
    tt.assert_close(denorm_state_no_norm, sample_no_norm['final_topo'])

# --- Tests for create_dataloaders ---

def test_create_dataloaders_basic(temp_pt_data):
    """Tests basic creation of dataloaders (no normalization)."""
    data_path, file_paths, _, _, _ = temp_pt_data
    batch_size = 2
    cfg = OmegaConf.create({
        'data': { # Data-specific parameters
            'processed_dir': data_path,
            'normalization': {'enabled': False},
            'train_split': 0.6, # Use train_split, val_split as expected by create_dataloaders
            'val_split': 0.2,
            'num_workers': 0,
            'seed': 42, # Seed might be better under training or top-level
        },
        'training': { # Training-specific parameters
            'batch_size': batch_size,
            # Add other training params if needed by create_dataloaders, though batch_size is the main one here
        }
        # Add other top-level keys like 'physics_params' if needed by the function being tested
    })

    # create_dataloaders returns a dictionary now
    dataloaders = create_dataloaders(cfg)
    assert isinstance(dataloaders, dict)
    assert 'train' in dataloaders and 'val' in dataloaders
    # assert 'test' in dataloaders # Removed check for test loader

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    # test_loader = dataloaders['test'] # Removed test loader

    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)
    # assert isinstance(test_loader, torch.utils.data.DataLoader) # Removed

    # Check batch size from config
    expected_batch_size = cfg['training']['batch_size'] # Access batch_size from config (Corrected path)
    assert train_loader.batch_size == expected_batch_size
    # Val loader might have different batch size or same, check config if specified, else assume same
    assert val_loader.batch_size == expected_batch_size
    # assert test_loader.batch_size == expected_batch_size # Removed

    # Check number of samples (approximate due to split)
    total_samples = len(file_paths)
    train_len = len(dataloaders['train'].dataset)
    val_len = len(dataloaders['val'].dataset)
    test_len = len(dataloaders['test'].dataset)
    assert train_len + val_len + test_len == total_samples
    # Check approximate split ratios (allow for rounding)
    # Check approximate split ratios (allow for rounding)
    expected_train_ratio = cfg.data.train_split
    expected_val_ratio = cfg.data.val_split
    expected_test_ratio = 1.0 - expected_train_ratio - expected_val_ratio
    # Use a slightly larger tolerance due to integer rounding of dataset sizes
    assert abs(train_len / total_samples - expected_train_ratio) < (2 / total_samples if total_samples > 0 else 0.1), f"Train split mismatch: expected ~{expected_train_ratio:.2f}, got {train_len / total_samples:.2f}"
    assert abs(val_len / total_samples - expected_val_ratio) < (2 / total_samples if total_samples > 0 else 0.1), f"Val split mismatch: expected ~{expected_val_ratio:.2f}, got {val_len / total_samples:.2f}"
    assert abs(test_len / total_samples - expected_test_ratio) < (2 / total_samples if total_samples > 0 else 0.1), f"Test split mismatch: expected ~{expected_test_ratio:.2f}, got {test_len / total_samples:.2f}"


def test_dataloader_batch_structure(temp_pt_data):
    """Tests the structure of a batch obtained from the dataloader (no normalization)."""
    data_path, file_paths, _, rows, cols = temp_pt_data
    batch_size = 2
    cfg = OmegaConf.create({
        'data': {
            'processed_dir': data_path,
            'normalization': {'enabled': False},
            'subset_fraction': 1.0,
            'seed': 42,
            'train_val_test_split': [1.0, 0.0, 0.0], # Use all for train loader
            'batch_size': batch_size,
            'num_workers': 0
        }
    })

    dataloaders = create_dataloaders(cfg)
    train_loader = dataloaders['train']
    batch = next(iter(train_loader)) # Get one batch

    assert isinstance(batch, dict)
    # Check for expected flat keys returned by the current FastscapeDataset
    expected_keys = ['initial_topo', 'final_topo', 'uplift_rate', 'k_f', 'k_d', 'm', 'n', 'run_time', 'target_shape']
    for key in expected_keys:
        assert key in batch, f"Expected key '{key}' not found in batch"
    # The test previously expected nested 'params'/'state', which is incorrect for the current dataset output.
    # The trainer is responsible for potentially restructuring this if needed by the model.
    # Coords might be added by the dataloader's collate_fn or later
    # assert 'coords' in batch

    # Check shapes and types based on the new structure
    assert isinstance(batch['initial_topo'], torch.Tensor)
    assert batch['initial_topo'].shape == (batch_size, 1, rows, cols)
    assert batch['initial_topo'].dtype == torch.float32

    assert isinstance(batch['final_topo'], torch.Tensor)
    assert batch['final_topo'].shape == (batch_size, 1, rows, cols)
    assert batch['final_topo'].dtype == torch.float32

    assert isinstance(batch['uplift_rate'], torch.Tensor)
    # Shape might be (B, H, W) or (B, 1, 1) depending on original data and collate_fn
    # Default collate adds batch dim. If original was (H,W), batch is (B, H, W)
    # If original was scalar, batch is (B,) - need to check collate behavior or fixture data
    # Assuming fixture saves spatial uplift_rate as (H, W)
    assert batch['uplift_rate'].shape == (batch_size, rows, cols)
    assert batch['uplift_rate'].dtype == torch.float32

    assert isinstance(batch['k_f'], torch.Tensor)
    assert batch['k_f'].shape == (batch_size,) # Batch of scalars
    assert batch['k_f'].dtype == torch.float32

    assert isinstance(batch['k_d'], torch.Tensor)
    assert batch['k_d'].shape == (batch_size, rows, cols)
    assert batch['k_d'].dtype == torch.float32

    assert isinstance(batch['m'], torch.Tensor)
    assert batch['m'].shape == (batch_size,)
    assert batch['m'].dtype == torch.float32

    assert isinstance(batch['n'], torch.Tensor)
    assert batch['n'].shape == (batch_size,)
    assert batch['n'].dtype == torch.float32

    assert isinstance(batch['run_time'], torch.Tensor)
    assert batch['run_time'].shape == (batch_size,)
    assert batch['run_time'].dtype == torch.float32

    assert 'target_shape' in batch # Check if key exists

    # Optional: Check coords if generated by dataloader
    # if 'coords' in batch:
    #     assert isinstance(batch['coords'], torch.Tensor)
    #     # Expected shape might be (B, N, 2) or (B, H, W, 2) depending on sampling strategy
    #     # assert batch['coords'].shape == ...
    #     assert batch['coords'].dtype == torch.float32
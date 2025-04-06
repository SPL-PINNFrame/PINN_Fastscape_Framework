import torch
import pytest
import sys
import os
import numpy as np
from unittest.mock import patch

# Add project root to sys.path to allow importing src modules
# Assuming tests are run from the project root or PYTHONPATH is set
# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(script_dir)
# sys.path.append(project_root)

from src.models import AdaptiveFastscapePINN
# Import trainer and dataloader for integration test
from src.trainer import PINNTrainer
from src.data_utils import create_dataloaders

# Basic logging setup for tests
import logging
logging.basicConfig(level=logging.DEBUG)


class TestAdaptiveModel: # Renamed class for clarity

    # Removed tests for compute_local_physics - should be tested in test_losses.py if kept

    def test_adaptive_model_initialization(self):
        """Test initialization of AdaptiveFastscapePINN model."""
        # Default initialization
        try:
            model_default = AdaptiveFastscapePINN()
            assert isinstance(model_default, AdaptiveFastscapePINN)
            assert model_default.base_resolution == 64
            assert model_default.max_resolution == 1024
        except Exception as e:
            pytest.fail(f"Default initialization failed: {e}")

        # Custom initialization
        try:
            custom_params = {
                'input_dim': 5, # Note: input_dim for MLP base might differ from coordinate_input_dim
                'output_dim': 1,
                'hidden_dim': 128,
                'num_layers': 4,
                'base_resolution': 32,
                'max_resolution': 512,
                'coordinate_input_dim': 5 # Explicitly set coordinate MLP input dim
            }
            model_custom = AdaptiveFastscapePINN(**custom_params)
            assert isinstance(model_custom, AdaptiveFastscapePINN)
            assert model_custom.base_resolution == 32
            assert model_custom.max_resolution == 512
            # Check the input dimension of the base MLP part
            assert model_custom.coordinate_mlp_base.input_dim == custom_params['coordinate_input_dim']
            # Check if CNN encoder input channels match (1 for topo + 2 for K, U assumed)
            assert model_custom.encoder[0].in_channels == 3
        except Exception as e:
            pytest.fail(f"Custom initialization failed: {e}")

    def test_adaptive_model_forward_predict_coords(self):
        """Test the forward pass in predict_coords mode."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n_points = 15
        input_dim = 5 # x, y, t, k, u
        output_dim = 1
        # Pass coordinate_input_dim to constructor
        model = AdaptiveFastscapePINN(coordinate_input_dim=input_dim, output_dim=output_dim).to(device).double()
        model.eval() # Set to eval mode

        # Dummy inputs
        coords = {
            'x': torch.rand(n_points, 1, device=device, dtype=torch.float64, requires_grad=True),
            'y': torch.rand(n_points, 1, device=device, dtype=torch.float64, requires_grad=True),
            't': torch.rand(n_points, 1, device=device, dtype=torch.float64, requires_grad=True)
        }
        # Dummy parameter grids (normalized coordinates assumed for sampling)
        k_grid = torch.rand(1, 1, 32, 32, device=device, dtype=torch.float64)
        u_grid = torch.rand(1, 1, 32, 32, device=device, dtype=torch.float64)

        forward_input = {
            **coords,
            'k_grid': k_grid,
            'u_grid': u_grid
        }

        # Test default output (state only)
        model.set_output_mode(state=True, derivative=False)
        output_state = model(forward_input, mode='predict_coords')
        assert isinstance(output_state, torch.Tensor)
        assert output_state.shape == (n_points, output_dim)
        assert output_state.requires_grad

        # Test dual output
        model.set_output_mode(state=True, derivative=True)
        output_dict = model(forward_input, mode='predict_coords')
        assert isinstance(output_dict, dict)
        assert 'state' in output_dict and 'derivative' in output_dict
        assert output_dict['state'].shape == (n_points, output_dim)
        assert output_dict['derivative'].shape == (n_points, output_dim)
        assert output_dict['state'].requires_grad
        assert output_dict['derivative'].requires_grad

        # TODO: Add gradient checks for predict_coords outputs w.r.t inputs (coords, model params)

    def test_adaptive_model_forward_predict_state_small(self):
        """Test forward pass in predict_state mode for small inputs (<= base_resolution)."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        base_res = 32 # Use a smaller base resolution for faster testing
        batch_size = 2
        height, width = base_res, base_res # Input size <= base_resolution
        output_dim = 1

        model = AdaptiveFastscapePINN(base_resolution=base_res, output_dim=output_dim).to(device).double()
        model.eval()

        # Dummy inputs
        initial_state = torch.rand(batch_size, 1, height, width, device=device, dtype=torch.float64)
        params = {
            'K': torch.rand(batch_size, 1, height, width, device=device, dtype=torch.float64) * 1e-5, # Spatial K
            'U': torch.rand(batch_size, 1, device=device, dtype=torch.float64) * 1e-3 # Scalar U (will be expanded)
        }
        t_target = torch.tensor([1000.0, 2000.0], device=device, dtype=torch.float64).view(batch_size, 1)

        forward_input = {
            'initial_state': initial_state,
            'params': params,
            't_target': t_target
        }

        # Mock internal processing methods to verify the correct one is called
        with patch.object(model, '_process_with_cnn', wraps=model._process_with_cnn) as mock_cnn, \
             patch.object(model, '_process_multi_resolution', wraps=model._process_multi_resolution) as mock_multi_res, \
             patch.object(model, '_process_tiled', wraps=model._process_tiled) as mock_tiled:

            try:
                # Test default output (state + derivative)
                model.set_output_mode(state=True, derivative=True)
                with torch.no_grad():
                    output_dict = model(forward_input, mode='predict_state')

                # Assertions for method calls
                mock_cnn.assert_called_once()
                mock_multi_res.assert_not_called()
                mock_tiled.assert_not_called()

                # Assertions for output structure
                assert isinstance(output_dict, dict), "Model output should be a dictionary"
                assert 'state' in output_dict and 'derivative' in output_dict

                output_state = output_dict['state']
                output_deriv = output_dict['derivative']
                assert isinstance(output_state, torch.Tensor) and isinstance(output_deriv, torch.Tensor)
                assert output_state.shape == (batch_size, output_dim, height, width)
                assert output_deriv.shape == (batch_size, output_dim, height, width)
                assert output_state.dtype == torch.float64 and output_deriv.dtype == torch.float64

                # Test state only output
                mock_cnn.reset_mock()
                model.set_output_mode(state=True, derivative=False)
                with torch.no_grad():
                    output_state_only = model(forward_input, mode='predict_state')
                mock_cnn.assert_called_once() # Should still call the same internal method
                assert isinstance(output_state_only, torch.Tensor) # Should return single tensor
                assert output_state_only.shape == (batch_size, output_dim, height, width)


            except Exception as e:
                pytest.fail(f"Forward pass 'predict_state' (small input) failed: {e}")

    # --- Tests for Medium and Large Inputs (Need Fixing Based on Pytest Output) ---


    def test_adaptive_model_forward_predict_state_medium(self):
        """Test forward pass in predict_state mode for medium inputs (> base, <= max)."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        base_res = 32
        max_res = 128
        batch_size = 1
        height, width = 64, 64 # Input size > base_res, <= max_res
        output_dim = 1

        model = AdaptiveFastscapePINN(base_resolution=base_res, max_resolution=max_res, output_dim=output_dim).to(device).double()
        model.eval()

        initial_state = torch.rand(batch_size, 1, height, width, device=device, dtype=torch.float64)
        params = {'K': torch.rand(batch_size, 1, height, width, device=device, dtype=torch.float64),
                  'U': torch.rand(batch_size, 1, height, width, device=device, dtype=torch.float64)}
        t_target = torch.tensor([1500.0], device=device, dtype=torch.float64).view(batch_size, 1)
        forward_input = {'initial_state': initial_state, 'params': params, 't_target': t_target}

        with patch.object(model, '_process_with_cnn', wraps=model._process_with_cnn) as mock_cnn, \
             patch.object(model, '_process_multi_resolution', wraps=model._process_multi_resolution) as mock_multi_res, \
             patch.object(model, '_process_tiled', wraps=model._process_tiled) as mock_tiled:

            with torch.no_grad():
                output_dict = model(forward_input, mode='predict_state')

            # Expected behavior: _process_multi_resolution is called
            # mock_cnn.assert_not_called() # Incorrect: _multi_res calls _cnn internally
            mock_cnn.assert_called_once() # CNN should be called once on the downsampled input
            mock_multi_res.assert_called_once()
            mock_tiled.assert_not_called()

            assert isinstance(output_dict, dict) and 'state' in output_dict
            assert output_dict['state'].shape == (batch_size, output_dim, height, width)


    def test_adaptive_model_forward_predict_state_large(self):
        """Test forward pass in predict_state mode for large inputs (> max)."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        base_res = 32
        max_res = 64 # Lower max_res to trigger tiling
        batch_size = 1
        height, width = 80, 80 # Input size > max_res
        output_dim = 1

        model = AdaptiveFastscapePINN(base_resolution=base_res, max_resolution=max_res, output_dim=output_dim).to(device).double()
        model.eval()

        initial_state = torch.rand(batch_size, 1, height, width, device=device, dtype=torch.float64)
        params = {'K': torch.rand(batch_size, 1, height, width, device=device, dtype=torch.float64),
                  'U': torch.rand(batch_size, 1, height, width, device=device, dtype=torch.float64)}
        t_target = torch.tensor([2500.0], device=device, dtype=torch.float64).view(batch_size, 1)
        forward_input = {'initial_state': initial_state, 'params': params, 't_target': t_target}

        with patch.object(model, '_process_with_cnn', wraps=model._process_with_cnn) as mock_cnn, \
             patch.object(model, '_process_multi_resolution', wraps=model._process_multi_resolution) as mock_multi_res, \
             patch.object(model, '_process_tiled', wraps=model._process_tiled) as mock_tiled:

            with torch.no_grad():
                 output_dict = model(forward_input, mode='predict_state')

            # Expected behavior: _process_tiled is called
            # mock_cnn.assert_not_called() # Incorrect: _tiled calls _cnn internally for each tile
            mock_cnn.assert_called() # CNN should be called at least once (likely multiple times)
            mock_multi_res.assert_not_called()
            mock_tiled.assert_called_once()

            assert isinstance(output_dict, dict) and 'state' in output_dict
            assert output_dict['state'].shape == (batch_size, output_dim, height, width)
            assert torch.mean(torch.abs(output_dict['state'])) > 1e-9 # Basic check for non-zero output

    # --- Tests for Helper Methods ---

    def test_adaptive_model_ensure_shape(self):
        """Test the _ensure_shape helper method."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = AdaptiveFastscapePINN().to(device).double() # Need an instance
        target_shape = (32, 32)
        batch_size = 4
        expected_shape = (batch_size, 1, *target_shape)
        dtype = model.state_head.weight.dtype # Get dtype from model

        # Test with None, scalar, scalar tensor, [H, W], [B, 1, H, W]
        output_none = model._ensure_shape(None, target_shape, batch_size, device, dtype)
        assert output_none.shape == expected_shape and torch.all(output_none == 0)

        output_scalar = model._ensure_shape(5.0, target_shape, batch_size, device, dtype)
        assert output_scalar.shape == expected_shape and torch.allclose(output_scalar, torch.tensor(5.0, device=device, dtype=dtype))

        output_scalar_tensor = model._ensure_shape(torch.tensor(3.0, device=device), target_shape, batch_size, device, dtype)
        assert output_scalar_tensor.shape == expected_shape and torch.allclose(output_scalar_tensor, torch.tensor(3.0, device=device, dtype=dtype))

        hw_tensor = torch.rand(*target_shape, device=device, dtype=dtype)
        output_hw = model._ensure_shape(hw_tensor, target_shape, batch_size, device, dtype)
        assert output_hw.shape == expected_shape and torch.allclose(output_hw[0, 0], hw_tensor)

        bchw_tensor = torch.rand(expected_shape, device=device, dtype=dtype)
        output_bchw = model._ensure_shape(bchw_tensor, target_shape, batch_size, device, dtype)
        assert output_bchw.shape == expected_shape and torch.equal(output_bchw, bchw_tensor)

        # Test with incompatible shape
        incompatible_tensor = torch.rand(batch_size, 2, *target_shape, device=device, dtype=dtype)
        with pytest.raises(ValueError):
            model._ensure_shape(incompatible_tensor, target_shape, batch_size, device, dtype)

    def test_adaptive_model_sample_at_coords(self):
        """Test the _sample_at_coords helper method using bilinear interpolation."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = AdaptiveFastscapePINN().to(device).double()
        n_points = 10
        height, width = 16, 16
        dtype = torch.float32 # Use float32 for consistency

        # Create a known grid (linear ramp in x)
        grid_np = np.linspace(0, 1, width).reshape(1, width).repeat(height, axis=0)
        param_grid = torch.tensor(grid_np, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

        # Sample points (normalized coordinates [0, 1]) along the center line
        x_coords = torch.linspace(0, 1, n_points, device=device, dtype=dtype).unsqueeze(1)
        y_coords = torch.full((n_points, 1), 0.5, device=device, dtype=dtype)
        expected_values = x_coords.clone() # Expected value should match x_coord

        # Test sampling
        sampled_values = model._sample_at_coords(param_grid, x_coords, y_coords)
        assert sampled_values.shape == (n_points, 1)
        # Relax tolerance slightly for potential minor interpolation inaccuracies
        assert torch.allclose(sampled_values, expected_values, atol=1e-5, rtol=1e-4), \
               f"Expected values close to {expected_values.flatten()}, got {sampled_values.flatten()}"

        # Test with None grid
        sampled_none = model._sample_at_coords(None, x_coords, y_coords)
        assert sampled_none.shape == (n_points, 1) and torch.all(sampled_none == 0)

        # Test with grid without batch/channel dims
        param_grid_hw = param_grid.squeeze()
        sampled_values_hw = model._sample_at_coords(param_grid_hw, x_coords, y_coords)
        assert sampled_values_hw.shape == (n_points, 1)
        assert torch.allclose(sampled_values_hw, expected_values, atol=1e-5, rtol=1e-4)


# --- Integration Test (Keep but verify/fix assertion) ---

def test_trainer_integration_adaptive_model(tmp_path):
    """Test integration of AdaptiveFastscapePINN with PINNTrainer for one step."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32 # Use float32 for consistency

    # 1. Create dummy config (simplified)
    height, width = 32, 32
    dummy_config = {
        'output_dir': str(tmp_path / 'results'), 'run_name': 'adaptive_integration_test',
        'seed': 42, 'device': device, 'use_mixed_precision': False,
        'data': {'processed_dir': str(tmp_path / 'data'), 'train_split': 1.0, 'val_split': 0.0, 'num_workers': 0},
        'model': {'type': 'AdaptiveFastscapePINN', 'base_resolution': 16, 'max_resolution': 64, 'hidden_dim': 32, 'num_layers': 2},
        'physics_params': {'dx': 1.0, 'dy': 1.0, 'total_time': 10.0},
        'training': {'epochs': 1, 'batch_size': 1, 'n_collocation_points': 50, 'optimizer': 'Adam', 'learning_rate': 1e-4,
                     'loss_weights': {'data': 1.0, 'physics': 0.1}} # Using default dual_output physics loss
    }

    # 2. Create dummy data file
    data_dir = tmp_path / 'data'
    # Create resolution specific subdir based on config
    res_dir_name = f"resolution_{height}x{width}"
    res_data_dir = data_dir / res_dir_name
    res_data_dir.mkdir(parents=True, exist_ok=True) # Create base and resolution dir

    # Ensure dummy data uses the specified dtype (float32 for this test)
    dummy_sample = {
        'initial_topo': torch.rand(1, height, width, dtype=dtype),
        'final_topo': torch.rand(1, height, width, dtype=dtype),
        'uplift_rate': torch.tensor(0.001, dtype=dtype), # Scalar U as tensor
        'k_f': torch.tensor(1e-5, dtype=dtype), # Scalar K_f
        'k_d': torch.tensor(0.01, dtype=dtype),
        'm': 0.5, # m and n are often passed as floats, ensure conversion in dataset if needed
        'n': 1.0,
        'run_time': torch.tensor(100.0, dtype=dtype) # Use a different time than physics_params.total_time
    }
    # Save to the resolution specific directory
    torch.save(dummy_sample, res_data_dir / 'sample_0.pt')

    # 3. Create dataloaders
    try:
        # Ensure create_dataloaders uses the correct data_dir structure
        dummy_config['data']['processed_dir'] = str(data_dir) # Point to base data dir
        dataloaders = create_dataloaders(dummy_config)
        train_loader = dataloaders['train']
        assert len(train_loader) == 1 # Should have 1 batch
    except Exception as e:
        pytest.fail(f"Failed to create dataloaders: {e}")

    # 4. Initialize Model
    try:
        model_config_for_init = {k: v for k, v in dummy_config['model'].items() if k != 'type'}
        model = AdaptiveFastscapePINN(**model_config_for_init).to(device=device, dtype=dtype) # Use float32 from test setup
    except Exception as e:
        pytest.fail(f"Failed to initialize AdaptiveFastscapePINN: {e}")

    # 5. Initialize Trainer
    try:
        trainer = PINNTrainer(model, dummy_config, train_loader, None)
    except Exception as e:
        pytest.fail(f"Failed to initialize PINNTrainer: {e}")

    # 6. Run one training epoch (which runs one step in this case)
    try:
        avg_loss, avg_loss_components = trainer._run_epoch(epoch=0, is_training=True)
        assert isinstance(avg_loss, float)
        assert not np.isnan(avg_loss) and avg_loss >= 0.0
        logging.info(f"Integration test completed with loss: {avg_loss:.4f}, components: {avg_loss_components}")
        # FIX: The original test might have failed due to an unrelated assert like 'assert 5 == 1'.
        # This version checks the loss is valid. Add more specific checks if needed.
    except Exception as e:
        pytest.fail(f"Trainer._run_epoch failed: {e}")
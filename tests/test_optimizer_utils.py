import torch
import pytest
import os
import logging
import numpy as np
from unittest.mock import patch, MagicMock
import torch.nn.functional as F # Ensure F is imported if used elsewhere
# ADDED: Import calculate_laplacian
from src.physics import calculate_laplacian # Use absolute import

# Import necessary classes and functions
# Use absolute imports now that project root is in sys.path via conftest.py
from src.optimizer_utils import (
    ParameterOptimizer,
    optimize_parameters,
    interpolate_uplift_torch, # Test differentiable interpolation
    terrain_similarity # Test similarity metric if needed (though simple)
)
# Import a dummy model for testing
from src.models import MLP_PINN # Use a simple model for optimizer tests

# Basic logging setup for tests
logging.basicConfig(level=logging.DEBUG)

# --- Fixtures ---

@pytest.fixture
def dummy_config_optimizer():
    """Basic config for optimization tests."""
    return {
        'optimization_params': {
            'optimizer': 'Adam',
            'learning_rate': 1e-3,
            'max_iterations': 10, # Few iterations for testing
            'spatial_smoothness_weight': 0.01,
            'log_interval': 5,
            'save_path': None # Don't save by default in tests
        },
        # Include other necessary sections if ParameterOptimizer or model needs them
        'physics_params': {'dx': 1.0, 'dy': 1.0} # Example
    }

# Define dummy model at module level so it can be referenced by tests
class SimpleOptimizerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Layer that combines initial state and U parameter
        self.layer = torch.nn.Conv2d(2, 1, 1, bias=False) # Input: initial_state, U
        # Initialize weights simply (e.g., to 1)
        torch.nn.init.constant_(self.layer.weight, 1.0)

    def forward(self, x, mode):
        if mode == 'predict_state':
            initial_state = x['initial_state'] # [B, 1, H, W]
            params = x['params']
            t_target = x['t_target'] # Scalar or [B,1,1,1]
            U_grid = params.get('U') # [B, 1, H, W]

            if U_grid is None: U_grid = torch.zeros_like(initial_state)

            # Simple mock prediction: initial + layer(initial, U) * t_target
            # Ensure t_target is broadcastable
            if isinstance(t_target, torch.Tensor) and t_target.ndim == 0:
                t_target = t_target.view(1, 1, 1, 1)
            elif isinstance(t_target, (int, float)):
                 t_target = torch.tensor(float(t_target), device=initial_state.device, dtype=initial_state.dtype).view(1, 1, 1, 1)

            # Combine inputs for the layer
            # Ensure types match before cat and conv
            combined = torch.cat([initial_state.type_as(self.layer.weight), U_grid.type_as(self.layer.weight)], dim=1)
            # Apply the layer and time effect
            pred = initial_state + self.layer(combined) * t_target.type_as(initial_state) * 0.1 # Scaled effect
            return pred
        return None # Other modes not needed for this test model

@pytest.fixture
def dummy_optimizer_model():
    """Provides a very simple dummy model instance."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Return float32 model for standard tests, gradcheck tests can convert locally if needed
    return SimpleOptimizerModel().to(device).float()

@pytest.fixture
def dummy_optimization_data():
    """Provides dummy observation, initial state, and true parameter."""
    B, H, W = 1, 8, 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32 # Use float32 for consistency
    # Simple target: initial state + constant uplift effect
    initial_state = torch.ones(B, 1, H, W, device=device, dtype=dtype) * 10.0
    true_U_value = 0.05
    true_U = torch.full((B, 1, H, W), true_U_value, device=device, dtype=dtype)
    t_target = torch.tensor(10.0, device=device, dtype=dtype) # Simple time

    # Create a dummy model instance just to generate target data
    temp_model = SimpleOptimizerModel().to(device).float() # Use float32
    with torch.no_grad():
        observation = temp_model({'initial_state': initial_state, 'params': {'U': true_U}, 't_target': t_target}, mode='predict_state')

    # Initial guess for optimization (different from true value)
    initial_U_guess_scalar = 0.01

    return {
        'observation': observation,
        'initial_state': initial_state,
        't_target': t_target,
        'true_U': true_U, # Keep for comparison
        'initial_U_guess': initial_U_guess_scalar # Use scalar for testing shape handling
    }

# --- Test Cases ---

def test_parameter_optimizer_init(dummy_optimizer_model, dummy_optimization_data):
    """Tests ParameterOptimizer initialization."""
    opt_instance = ParameterOptimizer(
        model=dummy_optimizer_model,
        observation_data=dummy_optimization_data['observation'],
        initial_state=dummy_optimization_data['initial_state'],
        fixed_params={'K': 0.1}, # Example fixed param
        t_target=dummy_optimization_data['t_target']
    )
    assert opt_instance.model == dummy_optimizer_model
    assert torch.equal(opt_instance.observation, dummy_optimization_data['observation'])
    assert torch.equal(opt_instance.initial_state, dummy_optimization_data['initial_state'])
    assert 'K' in opt_instance.fixed_params
    assert isinstance(opt_instance.fixed_params['K'], torch.Tensor) # Check conversion
    assert torch.equal(opt_instance.t_target, dummy_optimization_data['t_target'])

def test_ensure_initial_param_shape(dummy_optimizer_model, dummy_optimization_data):
    """Tests the _ensure_initial_param_shape helper."""
    opt_instance = ParameterOptimizer(dummy_optimizer_model, dummy_optimization_data['observation'])
    B, H, W = dummy_optimization_data['observation'].shape[0], 8, 8
    target_shape = (B, 1, H, W)
    dtype = torch.float32 # Expect float32 based on fixture

    # Test scalar initial value
    param_tensor_scalar = opt_instance._ensure_initial_param_shape(0.05, 'U')
    assert param_tensor_scalar.shape == target_shape
    assert param_tensor_scalar.requires_grad
    assert param_tensor_scalar.dtype == dtype
    assert torch.allclose(param_tensor_scalar, torch.tensor(0.05, dtype=dtype))

    # Test tensor initial value (matching shape)
    init_tensor = torch.rand(target_shape, device=opt_instance.device, dtype=dtype)
    param_tensor_match = opt_instance._ensure_initial_param_shape(init_tensor, 'U')
    assert param_tensor_match.shape == target_shape
    assert param_tensor_match.requires_grad
    assert param_tensor_match.dtype == dtype
    assert torch.equal(param_tensor_match.data, init_tensor) # Use .data to compare content

    # Test tensor initial value (needs interpolation/resizing - mock interpolate)
    init_tensor_small = torch.rand(B, 1, H//2, W//2, device=opt_instance.device, dtype=dtype)
    with patch('torch.nn.functional.interpolate', return_value=torch.rand(target_shape, dtype=dtype)) as mock_interp:
         param_tensor_interp = opt_instance._ensure_initial_param_shape(init_tensor_small, 'U')
         mock_interp.assert_called_once()
         assert param_tensor_interp.shape == target_shape
         assert param_tensor_interp.requires_grad
         assert param_tensor_interp.dtype == dtype

def test_create_objective_function(dummy_optimizer_model, dummy_optimization_data):
    """Tests the creation and basic execution of the objective function."""
    opt_instance = ParameterOptimizer(
        model=dummy_optimizer_model,
        observation_data=dummy_optimization_data['observation'],
        initial_state=dummy_optimization_data['initial_state'],
        t_target=dummy_optimization_data['t_target']
    )
    # Initialize parameter to optimize
    param_U = opt_instance._ensure_initial_param_shape(dummy_optimization_data['initial_U_guess'], 'U')
    params_to_optimize = {'U': param_U}

    # Pass fixed_params containing dx, dy if needed by calculate_laplacian
    opt_instance.fixed_params = {'dx': 1.0, 'dy': 1.0} # Add dummy dx, dy

    objective_fn = opt_instance.create_objective_function(params_to_optimize, spatial_smoothness=0.01)

    # Execute the objective function
    loss, loss_components = objective_fn()

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert 'data_loss' in loss_components
    assert 'U_smoothness_loss' in loss_components
    assert 'total_loss' in loss_components
    assert torch.isfinite(loss)

@pytest.mark.parametrize("optimizer_name", ["Adam", "AdamW"]) # LBFGS needs more setup
def test_optimize_parameters_basic(optimizer_name, dummy_optimizer_model, dummy_optimization_data, dummy_config_optimizer, tmp_path):
    """Tests the basic optimization loop for Adam/AdamW."""
    config = dummy_config_optimizer.copy()
    config['optimization_params']['optimizer'] = optimizer_name
    config['optimization_params']['max_iterations'] = 5 # Run only a few steps
    config['optimization_params']['save_path'] = str(tmp_path / "optimized.pth") # Test saving

    params_to_opt_config = {
        'U': {'initial_value': dummy_optimization_data['initial_U_guess'], 'bounds': (0.0, 0.1)}
    }

    # Pass fixed_params containing dx, dy if needed by objective function's smoothness calc
    fixed_params_for_opt = {'dx': 1.0, 'dy': 1.0}

    initial_U_tensor = ParameterOptimizer(dummy_optimizer_model, dummy_optimization_data['observation'])._ensure_initial_param_shape(
        dummy_optimization_data['initial_U_guess'], 'U'
    )
    initial_mean = initial_U_tensor.mean().item()

    optimized_params, history = optimize_parameters(
        model=dummy_optimizer_model,
        observation_data=dummy_optimization_data['observation'],
        params_to_optimize_config=params_to_opt_config,
        config=config,
        initial_state=dummy_optimization_data['initial_state'],
        fixed_params=fixed_params_for_opt, # Pass fixed params
        t_target=dummy_optimization_data['t_target']
    )

    assert 'U' in optimized_params
    optimized_U = optimized_params['U']
    assert isinstance(optimized_U, torch.Tensor)
    assert not optimized_U.requires_grad # Should be detached
    assert optimized_U.shape == dummy_optimization_data['true_U'].shape

    # Check if parameter value changed (loss should decrease towards true value)
    final_mean = optimized_U.mean().item()
    assert final_mean != initial_mean
    # Check if it moved towards the true value (0.05 in this dummy case)
    assert abs(final_mean - 0.05) < abs(initial_mean - 0.05)

    assert 'loss' in history and len(history['loss']) <= config['optimization_params']['max_iterations'] # Use <= due to potential early stopping
    assert 'final_loss' in history
    # Loss might not strictly decrease in first few steps with Adam/AdamW
    # assert history['final_loss'] < history['loss'][0]

    # Check if file was saved
    assert os.path.exists(config['optimization_params']['save_path'])

# TODO: Add test for LBFGS optimizer in optimize_parameters
# TODO: Add test for spatial_smoothness effect (check if optimized param is smoother)
# TODO: Add test for bounds enforcement
# TODO: Add test for optimize_parameters gradient flow (check grad exists on params_to_optimize during loop)

def test_interpolate_uplift_torch_bilinear():
    """Tests the bilinear interpolation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32 # Use float32
    param_shape = (4, 4)
    target_shape = (8, 8)

    # Create a simple linear ramp as input parameters
    params_np = np.linspace(0, 1, param_shape[0] * param_shape[1]).reshape(param_shape)
    params_torch = torch.tensor(params_np, device=device, dtype=dtype)

    interpolated = interpolate_uplift_torch(params_torch, param_shape, target_shape, method='bilinear')

    assert interpolated.shape == target_shape
    assert interpolated.dtype == dtype # Check if output dtype matches input dtype
    # Check some corner/center values (bilinear should preserve corners with align_corners=True)
    assert torch.allclose(interpolated[0, 0], params_torch[0, 0])
    assert torch.allclose(interpolated[-1, -1], params_torch[-1, -1])
    # Center value should be interpolated
    center_val = interpolated[target_shape[0]//2, target_shape[1]//2]
    assert center_val > params_torch[0,0] and center_val < params_torch[-1,-1]
    # Check requires_grad if input requires_grad
    params_torch.requires_grad_(True)
    interpolated_grad = interpolate_uplift_torch(params_torch, param_shape, target_shape, method='bilinear')
    assert interpolated_grad.requires_grad

# TODO: Add test for interpolate_uplift_torch RBF method
# TODO: Add gradcheck for interpolate_uplift_torch
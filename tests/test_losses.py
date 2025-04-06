import torch
import pytest
import logging
import numpy as np
from torch.autograd import gradcheck

# Import necessary functions/classes from src
# Use absolute imports now that project root is in sys.path via conftest.py
from src.losses import (
    compute_data_loss,
    compute_pde_residual, # Interpolation based
    # compute_pde_residual_adaptive, # Removed
    compute_pde_residual_grid_focused,
    compute_pde_residual_dual_output,
    # compute_local_physics, # Removed
    compute_smoothness_penalty,
    compute_total_loss,
    rbf_interpolate, # Keep if compute_pde_residual is kept
    sample_from_grid # Keep if needed elsewhere
)
# Import physics functions needed for some loss calculations or setup
from src.physics import calculate_dhdt_physics, calculate_slope_magnitude, calculate_laplacian
# Import a dummy model or use fixtures if defined in conftest.py
# from .conftest import DummyModel, DummyAdaptiveModel # Example

# Basic logging setup for tests
logging.basicConfig(level=logging.DEBUG)

# --- Fixtures (Consider moving complex ones to conftest.py) ---

@pytest.fixture
def dummy_config():
    """Provides a basic configuration dictionary."""
    return {
        'physics_params': {
            'U': 0.001, 'K_f': 1e-5, 'm': 0.5, 'n': 1.0, 'K_d': 0.01,
            'dx': 10.0, 'dy': 10.0, 'epsilon': 1e-10,
            'grid_height': 16, 'grid_width': 16,
            'domain_x': [0.0, 150.0], 'domain_y': [0.0, 150.0],
            'total_time': 1000.0,
            'drainage_area_kwargs': {'temp': 0.05, 'num_iters': 10},
            'rbf_sigma': 0.1, # For interpolation method
        },
        'training': {
            'loss_weights': {'data': 1.0, 'physics': 0.1, 'smoothness': 0.001}
        }
    }

@pytest.fixture
def dummy_grid_data(dummy_config):
    """Provides dummy grid data (prediction and target)."""
    params = dummy_config['physics_params']
    B, H, W = 1, params['grid_height'], params['grid_width']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Default to float32 for standard tests
    dtype = torch.float32
    pred = torch.rand(B, 1, H, W, device=device, dtype=dtype, requires_grad=True) * 100
    target = torch.rand(B, 1, H, W, device=device, dtype=dtype) * 100
    t_grid = torch.tensor(params['total_time'] / 2.0, device=device, dtype=dtype, requires_grad=True) # Example time
    return {'pred': pred, 'target': target, 't_grid': t_grid}

@pytest.fixture
def dummy_collocation_data(dummy_config):
    """Provides dummy collocation point data."""
    params = dummy_config['physics_params']
    N = 50 # Smaller number for faster gradcheck
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32 # Default to float32
    coords = {
        'x': torch.rand(N, 1, device=device, dtype=dtype, requires_grad=True) * params['domain_x'][1],
        'y': torch.rand(N, 1, device=device, dtype=dtype, requires_grad=True) * params['domain_y'][1],
        't': torch.rand(N, 1, device=device, dtype=dtype, requires_grad=True) * params['total_time']
    }
    # Simulate model prediction at collocation points
    h_pred = torch.rand(N, 1, device=device, dtype=dtype, requires_grad=True) * 100
    # Simulate dual output if needed
    dh_dt_pred = torch.rand(N, 1, device=device, dtype=dtype, requires_grad=True) * 0.1
    # Simulate parameter grids for sampling
    k_grid = torch.rand(1, 1, params['grid_height'], params['grid_width'], device=device, dtype=dtype) * 1e-5
    u_grid = torch.rand(1, 1, params['grid_height'], params['grid_width'], device=device, dtype=dtype) * 0.001
    coords['k_grid'] = k_grid
    coords['u_grid'] = u_grid
    return {'coords': coords, 'h_pred': h_pred, 'dh_dt_pred': dh_dt_pred}


# --- Test Cases ---

def test_compute_data_loss(dummy_grid_data):
    """Tests the basic data fidelity loss calculation."""
    loss = compute_data_loss(dummy_grid_data['pred'], dummy_grid_data['target'])
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad # Should be connected to prediction
    assert loss.item() >= 0

# --- PDE Residual Tests ---

# TODO: Add basic calculation tests for compute_pde_residual (Interpolation based)

# @pytest.mark.xfail(...) # Removed: Test passes, but see comments below
def test_gradcheck_pde_residual_interpolation(dummy_collocation_data, dummy_config):
    """Performs gradcheck on compute_pde_residual (interpolation method)."""
    h_pred = dummy_collocation_data['h_pred'].double().detach().requires_grad_(True)
    # Only check grad w.r.t time coordinate
    t_coords = dummy_collocation_data['coords']['t'].double().detach().requires_grad_(True)
    # Keep other coords fixed
    x_coords_fixed = dummy_collocation_data['coords']['x'].double().detach()
    y_coords_fixed = dummy_collocation_data['coords']['y'].double().detach()
    coords_fixed_spatial = {'x': x_coords_fixed, 'y': y_coords_fixed}

    physics_params = dummy_config['physics_params']

    # Check gradient w.r.t. h_pred
    is_correct_h = gradcheck(lambda h: compute_pde_residual(h, {**coords_fixed_spatial, 't': t_coords.detach()}, physics_params), (h_pred,), eps=1e-6, atol=1e-5, rtol=1e-3)
    assert is_correct_h, "Gradcheck failed for h_pred in interpolation PDE residual"

    # Check gradient w.r.t. t_coords
    is_correct_t = gradcheck(lambda t: compute_pde_residual(h_pred.detach(), {**coords_fixed_spatial, 't': t}, physics_params), (t_coords,), eps=1e-6, atol=1e-5, rtol=1e-3)
    assert is_correct_t, "Gradcheck failed for t_coords in interpolation PDE residual"


def test_compute_pde_residual_grid_focused(dummy_grid_data, dummy_config):
    """Tests the grid-focused PDE residual calculation."""
    loss = compute_pde_residual_grid_focused(
        dummy_grid_data['pred'],
        dummy_grid_data['t_grid'],
        dummy_config['physics_params']
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert loss.item() >= 0

# @pytest.mark.xfail(...) # Removed: Test passes (likely due to allow_unused=True), but see comments below
def test_gradcheck_pde_residual_grid_focused(dummy_grid_data, dummy_config):
    """Performs gradcheck on compute_pde_residual_grid_focused."""
    h_pred_grid = dummy_grid_data['pred'].double().detach().requires_grad_(True)
    t_grid = dummy_grid_data['t_grid'].double().detach().requires_grad_(True)
    physics_params = dummy_config['physics_params']

    # Check gradient w.r.t. h_pred_grid
    is_correct_h = gradcheck(lambda h: compute_pde_residual_grid_focused(h, t_grid.detach(), physics_params), (h_pred_grid,), eps=1e-6, atol=1e-5, rtol=1e-3)
    assert is_correct_h, "Gradcheck failed for h_pred_grid in grid_focused PDE residual"

    # Check gradient w.r.t. t_grid
    is_correct_t = gradcheck(lambda t: compute_pde_residual_grid_focused(h_pred_grid.detach(), t, physics_params), (t_grid,), eps=1e-6, atol=1e-5, rtol=1e-3)
    assert is_correct_t, "Gradcheck failed for t_grid in grid_focused PDE residual"


def test_compute_pde_residual_dual_output_grid(dummy_grid_data, dummy_config):
    """Tests the dual output PDE residual in grid mode."""
    outputs = {
        'state': dummy_grid_data['pred'],
        'derivative': torch.rand_like(dummy_grid_data['pred']) * 0.1 # Dummy predicted derivative
    }
    loss = compute_pde_residual_dual_output(outputs, dummy_config['physics_params'])
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert loss.item() >= 0

def test_gradcheck_pde_residual_dual_output_grid(dummy_grid_data, dummy_config):
    """Performs gradcheck on compute_pde_residual_dual_output in grid mode."""
    state = dummy_grid_data['pred'].double().detach().requires_grad_(True)
    derivative = (torch.rand_like(state) * 0.1).double().requires_grad_(True)
    outputs = {'state': state, 'derivative': derivative}
    physics_params = dummy_config['physics_params']

    # Check gradient w.r.t. state
    is_correct_state = gradcheck(lambda s: compute_pde_residual_dual_output({'state': s, 'derivative': derivative.detach()}, physics_params), (state,), eps=1e-6, atol=1e-5, rtol=1e-3)
    assert is_correct_state, "Gradcheck failed for state input in dual_output PDE residual (grid)"

    # Check gradient w.r.t. derivative
    is_correct_deriv = gradcheck(lambda d: compute_pde_residual_dual_output({'state': state.detach(), 'derivative': d}, physics_params), (derivative,), eps=1e-6, atol=1e-5, rtol=1e-3)
    assert is_correct_deriv, "Gradcheck failed for derivative input in dual_output PDE residual (grid)"


def test_compute_pde_residual_dual_output_coords_disabled(dummy_collocation_data, dummy_config):
    """Tests that dual output PDE residual raises error or warns in coordinate mode."""
    outputs = {
        'state': dummy_collocation_data['h_pred'],
        'derivative': dummy_collocation_data['dh_dt_pred'],
        'coords': dummy_collocation_data['coords']
    }
    physics_params_with_coords = dummy_config['physics_params'].copy()
    physics_params_with_coords['coords'] = dummy_collocation_data['coords']
    # Add grids needed for sampling if that part remained
    physics_params_with_coords['k_grid'] = dummy_collocation_data['coords']['k_grid']
    physics_params_with_coords['u_grid'] = dummy_collocation_data['coords']['u_grid']

    # Check if it returns zero loss and logs a warning, as per current implementation in src/losses.py
    # The function was modified to log an error and return zero loss.
    # We check for zero loss here. Checking logs requires the 'caplog' fixture from pytest.
    loss = compute_pde_residual_dual_output(outputs, physics_params_with_coords)
    assert isinstance(loss, torch.Tensor)
    # The returned tensor might still require grad depending on how zero_like was created
    # assert not loss.requires_grad
    # Use assert_close for robust floating point comparison against zero
    torch.testing.assert_close(loss, torch.tensor(0.0, device=loss.device, dtype=loss.dtype), atol=1e-8, rtol=0, msg="Expected near-zero loss for dual_output in coordinate mode")
    # assert loss.item() == 0.0, "Expected zero loss for dual_output in coordinate mode due to implementation issues" # Replaced with assert_close
    # Optional: Add check for warning log message using caplog fixture if needed


# --- Other Loss Tests ---

def test_compute_smoothness_penalty(dummy_grid_data, dummy_config):
    """Tests the smoothness penalty calculation."""
    params = dummy_config['physics_params']
    # Ensure input is float32 for this test, as gradcheck is not involved
    pred_float32 = dummy_grid_data['pred'].float().requires_grad_(True)
    loss = compute_smoothness_penalty(pred_float32, params['dx'], params['dy'])
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert loss.item() >= 0

def test_compute_total_loss(dummy_grid_data, dummy_config):
    """Tests the calculation of the total weighted loss."""
    # Ensure inputs are float32
    pred_float32 = dummy_grid_data['pred'].float().requires_grad_(True)
    target_float32 = dummy_grid_data['target'].float()
    dummy_physics_loss = torch.tensor(0.5, device=pred_float32.device, dtype=torch.float32, requires_grad=True)

    total_loss, loss_components = compute_total_loss(
        data_pred=pred_float32,
        final_topo=target_float32,
        physics_loss_value=dummy_physics_loss,
        physics_params=dummy_config['physics_params'],
        loss_weights=dummy_config['training']['loss_weights']
    )
    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.requires_grad
    assert 'data_loss' in loss_components
    assert 'physics_loss' in loss_components
    assert 'smoothness_loss' in loss_components
    assert 'total_loss' in loss_components
    assert torch.isfinite(total_loss)

def test_compute_total_loss_no_physics(dummy_grid_data, dummy_config):
    """Tests total loss when physics loss is None or weight is zero."""
    # Ensure inputs are float32
    pred_float32 = dummy_grid_data['pred'].float().requires_grad_(True)
    target_float32 = dummy_grid_data['target'].float()
    weights_no_physics = dummy_config['training']['loss_weights'].copy()
    weights_no_physics['physics'] = 0.0

    total_loss, loss_components = compute_total_loss(
        data_pred=pred_float32,
        final_topo=target_float32,
        physics_loss_value=None, # Pass None
        physics_params=dummy_config['physics_params'],
        loss_weights=weights_no_physics
    )
    assert isinstance(total_loss, torch.Tensor)
    assert loss_components['physics_loss'] == 0.0 # Check that physics component is zero

# Remember gradcheck can be slow and memory intensive. Use smaller inputs if needed.
# Ensure double precision (dtype=torch.float64) for inputs requiring grad.
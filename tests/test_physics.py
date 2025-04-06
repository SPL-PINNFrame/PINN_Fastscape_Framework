import pytest
import torch
import torch.testing as tt # Keep for assert_close if needed later

# Assuming the functions are in src.physics
# Adjust the import path if necessary
from src.physics import calculate_drainage_area_differentiable_optimized, calculate_dhdt_physics
# Import custom derivatives if needed for direct comparison/testing, though calculate_dhdt_physics uses them internally
from src.derivatives import laplacian as custom_laplacian

# Helper function to create simple terrains
def create_simple_terrain(rows, cols, slope_x=0.01, slope_y=0.01, base_elev=0.0, dtype=torch.float32, device='cpu'):
    """Creates a simple terrain grid."""
    y_coords, x_coords = torch.meshgrid(
        torch.arange(rows, dtype=dtype, device=device),
        torch.arange(cols, dtype=dtype, device=device),
        indexing='ij'
    )
    terrain = base_elev + slope_y * y_coords + slope_x * x_coords
    return terrain.unsqueeze(0).unsqueeze(0) # Add batch and channel dims (B, C, H, W)

# --- Tests for calculate_drainage_area_differentiable_optimized ---

@pytest.mark.parametrize("dtype", [torch.float32]) # float64 can be slow/unstable with softmax/iterative methods
@pytest.mark.parametrize("temp", [0.01, 0.1]) # Test different levels of flow concentration
def test_drainage_area_simple_slope(dtype, temp):
    """Tests drainage area calculation on a simple linear slope."""
    rows, cols = 7, 7 # Use slightly larger grid
    dx, dy = 1.0, 1.0
    precip = 1.0
    num_iters = 10 # Sufficient iterations for small grid
    cell_area = dx * dy

    # Create terrain sloping towards bottom-right corner
    elev = create_simple_terrain(rows, cols, slope_x=0.01, slope_y=0.01, dtype=dtype)

    drainage_area = calculate_drainage_area_differentiable_optimized(
        elev, dx=dx, dy=dy, precip=precip, temp=temp, num_iters=num_iters
    )

    assert drainage_area.shape == elev.shape
    assert drainage_area.dtype == dtype
    # Check bounds: Area should be at least local precipitation * cell_area
    assert (drainage_area >= (precip * cell_area - 1e-6)).all(), "Drainage area should be >= local input"

    # Check general trend: Area should increase towards the outlet (bottom-right)
    # Top-left corner should have area close to local input
    tt.assert_close(drainage_area[0, 0, 0, 0], torch.tensor(precip * cell_area, dtype=dtype), atol=1e-3, rtol=1e-3)
    # Bottom-right corner should have accumulated area (roughly rows*cols*precip*cell_area for single outlet)
    # Due to softmax distribution, it might be less than full accumulation
    expected_max_area = rows * cols * precip * cell_area
    # Assert it's significantly larger than single cell area but not necessarily the theoretical max
    assert drainage_area[0, 0, -1, -1] > precip * cell_area * 2
    assert drainage_area[0, 0, -1, -1] <= expected_max_area

    # Check middle cell has more area than corner cell
    assert drainage_area[0, 0, rows // 2, cols // 2] > drainage_area[0, 0, 0, 0] * 1.1 # Ensure some accumulation

    # Check for NaNs
    assert not torch.isnan(drainage_area).any()

@pytest.mark.parametrize("dtype", [torch.float32])
def test_drainage_area_pit(dtype):
    """Tests drainage area calculation with a pit."""
    rows, cols = 5, 5
    dx, dy = 1.0, 1.0
    precip = 1.0
    temp = 0.01 # Lower temp concentrates flow more
    num_iters = 10
    cell_area = dx * dy
    pit_row, pit_col = rows // 2, cols // 2

    elev = create_simple_terrain(rows, cols, slope_x=0, slope_y=0, base_elev=1.0, dtype=dtype)
    # Create a pit in the center
    elev[0, 0, pit_row, pit_col] = 0.0

    drainage_area = calculate_drainage_area_differentiable_optimized(
        elev, dx=dx, dy=dy, precip=precip, temp=temp, num_iters=num_iters
    )

    assert drainage_area.shape == elev.shape
    assert drainage_area.dtype == dtype
    assert (drainage_area >= (precip * cell_area - 1e-6)).all(), "Drainage area should be >= local input"

    # Expect the pit cell to have a high drainage area
    # It should receive flow from most/all other cells in this simple case
    expected_pit_area = rows * cols * precip * cell_area
    # Check it's significantly higher than others and close to total area
    assert drainage_area[0, 0, pit_row, pit_col] > expected_pit_area * 0.8 # Allow for some distribution loss/edge effects
    assert drainage_area[0, 0, pit_row, pit_col] <= expected_pit_area

    # Cells flowing into the pit (e.g., immediate neighbors) should have lower area, close to local input
    tt.assert_close(drainage_area[0, 0, pit_row + 1, pit_col], torch.tensor(precip * cell_area, dtype=dtype), atol=1e-3, rtol=1e-3)
    tt.assert_close(drainage_area[0, 0, pit_row, pit_col + 1], torch.tensor(precip * cell_area, dtype=dtype), atol=1e-3, rtol=1e-3)

    # Check for NaNs
    assert not torch.isnan(drainage_area).any()

@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("temp", [0.1, 1.0]) # Higher temp disperses more on flat
def test_drainage_area_flat(dtype, temp):
    """Tests drainage area calculation on a flat surface."""
    rows, cols = 5, 5
    dx, dy = 1.0, 1.0
    precip = 1.0
    num_iters = 20 # More iterations might be needed for flat areas to stabilize
    cell_area = dx * dy

    elev = torch.zeros((1, 1, rows, cols), dtype=dtype) # Perfectly flat

    drainage_area = calculate_drainage_area_differentiable_optimized(
        elev, dx=dx, dy=dy, precip=precip, temp=temp, num_iters=num_iters
    )

    assert drainage_area.shape == elev.shape
    assert drainage_area.dtype == dtype
    assert (drainage_area >= (precip * cell_area - 1e-6)).all(), "Drainage area should be >= local input"

    # Behavior on flat surfaces: Softmax with zero slopes gives equal weights (1/8).
    # After iterations, area should ideally remain close to the local input value everywhere,
    # as flow disperses outwards and inwards roughly equally.
    # Check that the area is close to the expected local input across the grid.
    expected_area = torch.full_like(elev, precip * cell_area)
    # Use a larger tolerance for flat areas as iterative methods might have edge effects
    tt.assert_close(drainage_area, expected_area, atol=1e-2, rtol=1e-2)

    # Check for NaNs
    assert not torch.isnan(drainage_area).any()

# TODO: Add test_drainage_area_boundary_conditions if specific boundary handling is expected

# --- Tests for calculate_dhdt_physics (replaces compute_local_physics tests) ---

@pytest.mark.parametrize("dtype", [torch.float32])
def test_calculate_dhdt_pure_diffusion(dtype):
    """Tests calculate_dhdt_physics with only diffusion active."""
    rows, cols = 5, 5
    dx, dy = 1.0, 1.0
    K_d = 1e-3
    # Create a simple parabolic hill for non-zero Laplacian
    y, x = torch.meshgrid(torch.arange(rows, dtype=dtype) - rows // 2,
                          torch.arange(cols, dtype=dtype) - cols // 2, indexing='ij')
    h = (-(x**2 + y**2) * 0.01).unsqueeze(0).unsqueeze(0) # (B, C, H, W)

    # Parameters for calculate_dhdt_physics
    params = {
        'h': h,
        'U': torch.tensor(0.0, dtype=dtype),      # No uplift
        'K_f': torch.tensor(0.0, dtype=dtype),   # Disable erosion
        'm': torch.tensor(0.5, dtype=dtype),
        'n': torch.tensor(1.0, dtype=dtype),
        'K_d': torch.tensor(K_d, dtype=dtype), # Enable diffusion
        'dx': dx,
        'dy': dy,
        'precip': 0.0, # No precip needed if K_f=0
        'da_optimize_params': {'temp': 0.01, 'num_iters': 1} # DA params needed but won't affect result if K_f=0
    }

    dhdt = calculate_dhdt_physics(**params)

    # Calculate expected dhdt = K_d * Laplacian(h) using the same custom laplacian
    # Need to handle padding similarly if comparing numerically
    # For simplicity, just check shape, dtype, and sign (diffusion lowers peaks, raises valleys)
    expected_laplacian = custom_laplacian(h, dx, dy)
    expected_dhdt = K_d * expected_laplacian

    assert dhdt.shape == h.shape
    assert dhdt.dtype == dtype
    # Check that the center (peak) has negative dhdt (erosion by diffusion)
    assert dhdt[0, 0, rows // 2, cols // 2] < 0
    # Check corners (valleys) have positive dhdt (filling by diffusion)
    assert dhdt[0, 0, 0, 0] > 0
    # Compare numerically with expected result
    tt.assert_close(dhdt, expected_dhdt, atol=1e-6, rtol=1e-5)
    # Check for NaNs
    assert not torch.isnan(dhdt).any()
    # Basic boundary check (values should not be extreme outliers)
    assert torch.isfinite(dhdt[:, :, 0, :]).all() # Top boundary
    assert torch.isfinite(dhdt[:, :, -1, :]).all() # Bottom boundary
    assert torch.isfinite(dhdt[:, :, :, 0]).all() # Left boundary
    assert torch.isfinite(dhdt[:, :, :, -1]).all() # Right boundary


@pytest.mark.parametrize("dtype", [torch.float32])
def test_calculate_dhdt_pure_erosion_uplift(dtype):
    """Tests calculate_dhdt_physics with only erosion and uplift active."""
    rows, cols = 7, 7
    dx, dy = 1.0, 1.0
    U = 1e-4
    K_f = 1e-5
    m = 0.5
    n = 1.0
    precip = 1.0
    temp = 0.01
    num_iters = 10

    h = create_simple_terrain(rows, cols, slope_x=0.01, slope_y=0.01, dtype=dtype)

    params = {
        'h': h,
        'U': torch.tensor(U, dtype=dtype),
        'K_f': torch.tensor(K_f, dtype=dtype),
        'm': torch.tensor(m, dtype=dtype),
        'n': torch.tensor(n, dtype=dtype),
        'K_d': torch.tensor(0.0, dtype=dtype), # Disable diffusion
        'dx': dx,
        'dy': dy,
        'precip': precip,
        'da_optimize_params': {'temp': temp, 'num_iters': num_iters}
    }

    dhdt = calculate_dhdt_physics(**params)

    assert dhdt.shape == h.shape
    assert dhdt.dtype == dtype

    # Basic checks:
    # - Where slope/area are low (e.g., divides/corners), dhdt should be close to U.
    # - Where slope/area are high (e.g., outlet), dhdt should be U - Erosion < U.

    # Check corner (low slope/area)
    # Erosion term E = K_f * A^m * S^n should be small here
    # A is close to cell_area, S is the constant slope
    # We expect dhdt ~ U
    # Need slope calculation from custom derivative
    from src.derivatives import spatial_gradient # Use absolute import
    dh_dx = spatial_gradient(h, dim=1, spacing=dx)
    dh_dy = spatial_gradient(h, dim=0, spacing=dy)
    slope_mag = torch.sqrt(dh_dx**2 + dh_dy**2 + 1e-10)
    corner_slope = slope_mag[0, 0, 0, 0]
    corner_area = dx * dy * precip # Approx area at corner
    corner_erosion = K_f * (corner_area**m) * (corner_slope**n)
    expected_corner_dhdt = U - corner_erosion
    # Use larger tolerance due to DA approximation and derivative edge effects
    tt.assert_close(dhdt[0, 0, 0, 0], expected_corner_dhdt, atol=U*0.5, rtol=0.5) # Check if it's roughly U

    # Check outlet (high slope/area)
    # Erosion term should be significant, dhdt < U
    assert dhdt[0, 0, -1, -1] < U

    # Check for NaNs
    assert not torch.isnan(dhdt).any()
    # Basic boundary check
    assert torch.isfinite(dhdt[:, :, 0, :]).all()
    assert torch.isfinite(dhdt[:, :, -1, :]).all()
    assert torch.isfinite(dhdt[:, :, :, 0]).all()
    assert torch.isfinite(dhdt[:, :, :, -1]).all()


@pytest.mark.parametrize("dtype", [torch.float32])
def test_calculate_dhdt_combined(dtype):
    """Tests calculate_dhdt_physics with erosion, diffusion, and uplift."""
    rows, cols = 7, 7
    dx, dy = 1.0, 1.0
    U = 1e-4
    K_f = 1e-5
    m = 0.5
    n = 1.0
    K_d = 1e-3
    precip = 1.0
    temp = 0.01
    num_iters = 10

    h = create_simple_terrain(rows, cols, slope_x=0.01, slope_y=0.01, dtype=dtype)

    params = {
        'h': h,
        'U': torch.tensor(U, dtype=dtype),
        'K_f': torch.tensor(K_f, dtype=dtype),
        'm': torch.tensor(m, dtype=dtype),
        'n': torch.tensor(n, dtype=dtype),
        'K_d': torch.tensor(K_d, dtype=dtype),
        'dx': dx,
        'dy': dy,
        'precip': precip,
        'da_optimize_params': {'temp': temp, 'num_iters': num_iters}
    }

    dhdt = calculate_dhdt_physics(**params)

    assert dhdt.shape == h.shape
    assert dhdt.dtype == dtype
    # Check for NaNs or Infs as a basic sanity check for combined effects
    assert not torch.isnan(dhdt).any()
    assert not torch.isinf(dhdt).any()
    # Further numerical checks are complex, rely on component tests.
    # Basic boundary check
    assert torch.isfinite(dhdt[:, :, 0, :]).all()
    assert torch.isfinite(dhdt[:, :, -1, :]).all()
    assert torch.isfinite(dhdt[:, :, :, 0]).all()
    assert torch.isfinite(dhdt[:, :, :, -1]).all()

@pytest.mark.skip(reason="Boundary condition test requires specific terrain setup and analysis.")
def test_calculate_dhdt_boundary_conditions(dtype=torch.float32):
    """Tests calculate_dhdt_physics specifically focusing on boundary behavior."""
    # TODO: Implement test with terrain designed to test boundary conditions
    # e.g., steep slope towards one boundary, flat area near another.
    # Check if dhdt values at boundary cells are consistent with padding mode ('replicate' used in physics.py)
    # and physical expectations (e.g., no flux implied by replicate padding for diffusion).
    pass
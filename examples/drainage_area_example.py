"""
Example script demonstrating the enhanced drainage area calculation.
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Import the drainage area functions
from src.physics import calculate_drainage_area_differentiable_optimized
try:
    from src.drainage_area_enhanced import calculate_drainage_area_enhanced
    HAS_ENHANCED_DRAINAGE = True
except ImportError:
    HAS_ENHANCED_DRAINAGE = False
    print("Enhanced drainage area calculation not available.")

def create_test_dem(size=64, device='cpu', dtype=torch.float32):
    """
    Creates a test DEM with various features for testing drainage area calculation.
    """
    # Create a grid of coordinates
    y, x = torch.meshgrid(
        torch.arange(size, dtype=dtype, device=device),
        torch.arange(size, dtype=dtype, device=device),
        indexing='ij'
    )

    # Create a sloped surface
    dem = 100.0 - 0.5 * x - 0.2 * y

    # Add some noise
    noise = torch.randn(size, size, device=device, dtype=dtype) * 0.2
    dem = dem + noise

    # Add a depression (pit)
    center_x, center_y = size // 3, size // 3
    radius_sq = (size // 10) ** 2
    dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
    depression_depth = 10.0
    dem = dem - depression_depth * torch.exp(-dist_sq / (2 * radius_sq)) * (dist_sq < radius_sq).float()

    # Add a ridge
    ridge_x = 2 * size // 3
    ridge_width = size // 20
    ridge_height = 5.0
    dem = dem + ridge_height * torch.exp(-(x - ridge_x) ** 2 / (2 * ridge_width ** 2))

    # Reshape to (1, 1, size, size) for the model
    return dem.unsqueeze(0).unsqueeze(0)

def main():
    """Main function to demonstrate drainage area calculation."""
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create test DEM
    size = 64
    print(f"Creating test DEM of size {size}x{size}...")
    dem = create_test_dem(size=size, device=device)

    # Parameters
    dx = 10.0
    dy = 10.0
    precip = 1.0

    # Calculate drainage area using original method
    print("\n--- Running Original Method ---")
    start_time = time.time()
    original_params = {
        'temp': 0.01,
        'num_iters': 50,
        'verbose': True
    }
    original_result = calculate_drainage_area_differentiable_optimized(
        dem, dx=dx, dy=dy, precip=precip, **original_params
    )
    original_time = time.time() - start_time
    print(f"Original method completed in {original_time:.3f} seconds")

    # Calculate drainage area using enhanced method if available
    if HAS_ENHANCED_DRAINAGE:
        print("\n--- Running Enhanced Method ---")
        start_time = time.time()
        enhanced_params = {
            'initial_temp': 0.1,
            'end_temp': 1e-3,
            'annealing_factor': 0.99,
            'max_iters': 20,
            'lambda_dir': 1.0,
            'convergence_threshold': 1e-3,
            'special_depression_handling': True,
            'verbose': True,
            'stable_mode': True
        }
        enhanced_result = calculate_drainage_area_enhanced(
            dem, dx=dx, dy=dy, precip=precip, **enhanced_params
        )
        enhanced_time = time.time() - start_time
        print(f"Enhanced method completed in {enhanced_time:.3f} seconds")

        # Calculate statistics
        abs_diff = torch.abs(enhanced_result - original_result)
        rel_diff = abs_diff / (original_result + 1e-10)

        max_abs_diff = torch.max(abs_diff).item()
        mean_abs_diff = torch.mean(abs_diff).item()
        max_rel_diff = torch.max(rel_diff).item()
        mean_rel_diff = torch.mean(rel_diff).item()

        print("\n--- Comparison Statistics ---")
        print(f"Max Absolute Difference: {max_abs_diff:.4f}")
        print(f"Mean Absolute Difference: {mean_abs_diff:.4f}")
        print(f"Max Relative Difference: {max_rel_diff:.4f}")
        print(f"Mean Relative Difference: {mean_rel_diff:.4f}")

        # Check mass conservation
        total_precip = torch.sum(torch.ones_like(dem) * precip * dx * dy).item()
        total_original = torch.sum(original_result).item()
        total_enhanced = torch.sum(enhanced_result).item()

        print("\n--- Mass Conservation ---")
        print(f"Total Precipitation: {total_precip:.4f}")
        print(f"Total Original: {total_original:.4f}")
        print(f"Total Enhanced: {total_enhanced:.4f}")
        print(f"Original Mass Error: {(total_original - total_precip) / total_precip:.4%}")
        print(f"Enhanced Mass Error: {(total_enhanced - total_precip) / total_precip:.4%}")

        # Visualize results
        try:
            plt.figure(figsize=(15, 10))

            # Plot DEM
            plt.subplot(2, 3, 1)
            plt.imshow(dem[0, 0].cpu().numpy(), cmap='terrain')
            plt.title('DEM')
            plt.colorbar(label='Elevation')

            # Plot original result
            plt.subplot(2, 3, 2)
            plt.imshow(np.log1p(original_result[0, 0].cpu().numpy()), cmap='Blues')
            plt.title('Original Drainage Area (log1p)')
            plt.colorbar(label='log(1 + Area)')

            # Plot enhanced result
            plt.subplot(2, 3, 3)
            plt.imshow(np.log1p(enhanced_result[0, 0].cpu().numpy()), cmap='Blues')
            plt.title('Enhanced Drainage Area (log1p)')
            plt.colorbar(label='log(1 + Area)')

            # Plot absolute difference
            plt.subplot(2, 3, 4)
            plt.imshow(abs_diff[0, 0].cpu().numpy(), cmap='Reds')
            plt.title('Absolute Difference')
            plt.colorbar(label='|Enhanced - Original|')

            # Plot relative difference
            plt.subplot(2, 3, 5)
            rel_diff_np = rel_diff[0, 0].cpu().numpy()
            rel_diff_np = np.clip(rel_diff_np, 0, 1)  # Clip for better visualization
            plt.imshow(rel_diff_np, cmap='Reds', vmax=0.5)
            plt.title('Relative Difference (clipped to 0-1)')
            plt.colorbar(label='|Enhanced - Original| / Original')

            # Save figure
            plt.tight_layout()
            plt.savefig('drainage_area_comparison.png', dpi=150)
            plt.show()

        except Exception as e:
            print(f"Error during visualization: {e}")

    print("\nDone!")

if __name__ == "__main__":
    main()

"""
Test script for comparing the original and enhanced drainage area calculations.
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.physics import calculate_drainage_area_differentiable_optimized
from src.drainage_area_enhanced import calculate_drainage_area_enhanced

def create_test_dem(size=64, device='cpu', dtype=torch.float32):
    """
    Creates a test DEM with various features for testing drainage area calculation.

    Args:
        size: Size of the DEM (size x size)
        device: Device to create the DEM on
        dtype: Data type of the DEM

    Returns:
        Tensor of shape (1, 1, size, size) containing the DEM
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

    # Add a flat area
    flat_x_min, flat_x_max = 3 * size // 4, 7 * size // 8
    flat_y_min, flat_y_max = size // 8, size // 4
    flat_mask = ((x >= flat_x_min) & (x <= flat_x_max) &
                 (y >= flat_y_min) & (y <= flat_y_max))
    flat_value = dem[flat_y_min, flat_x_min]
    dem = torch.where(flat_mask, flat_value, dem)

    # Reshape to (1, 1, size, size) for the model
    return dem.unsqueeze(0).unsqueeze(0)

def run_comparison(dem, dx=10.0, dy=10.0, precip=1.0):
    """
    Runs both drainage area calculations and compares them.

    Args:
        dem: DEM tensor of shape (1, 1, H, W)
        dx: Grid spacing in x direction
        dy: Grid spacing in y direction
        precip: Precipitation rate

    Returns:
        Dictionary containing results and timing information
    """
    device = dem.device
    dtype = dem.dtype

    # Parameters for both methods
    common_params = {
        'dx': dx,
        'dy': dy,
        'precip': precip
    }

    # Original method parameters
    original_params = {
        'temp': 0.01,
        'num_iters': 50,
        'verbose': True
    }

    # Enhanced method parameters
    enhanced_params = {
        'initial_temp': 0.05,
        'end_temp': 1e-4,
        'annealing_factor': 0.99,
        'max_iters': 30,
        'lambda_dir': 1.0,
        'convergence_threshold': 1e-4,
        'special_depression_handling': True,
        'verbose': True,
        'check_mass_conservation': True,
        'stable_mode': True  # Use stable mode
    }

    # Run original method
    print("\n--- Running Original Method ---")
    start_time = time.time()
    try:
        original_result = calculate_drainage_area_differentiable_optimized(
            dem, **common_params, **original_params
        )
        original_time = time.time() - start_time
        original_success = True
        print(f"Original method completed in {original_time:.3f} seconds")
    except Exception as e:
        print(f"Original method failed: {e}")
        original_result = torch.zeros_like(dem)
        original_time = time.time() - start_time
        original_success = False

    # Run enhanced method
    print("\n--- Running Enhanced Method ---")
    start_time = time.time()
    try:
        enhanced_result = calculate_drainage_area_enhanced(
            dem, **common_params, **enhanced_params
        )
        enhanced_time = time.time() - start_time
        enhanced_success = True
        print(f"Enhanced method completed in {enhanced_time:.3f} seconds")
    except Exception as e:
        print(f"Enhanced method failed: {e}")
        enhanced_result = torch.zeros_like(dem)
        enhanced_time = time.time() - start_time
        enhanced_success = False

    # Calculate statistics if both methods succeeded
    if original_success and enhanced_success:
        # Calculate difference
        abs_diff = torch.abs(enhanced_result - original_result)
        rel_diff = abs_diff / (original_result + 1e-10)

        # Calculate statistics
        max_abs_diff = torch.max(abs_diff).item()
        mean_abs_diff = torch.mean(abs_diff).item()
        max_rel_diff = torch.max(rel_diff).item()
        mean_rel_diff = torch.mean(rel_diff).item()

        # Calculate correlation
        original_flat = original_result.flatten()
        enhanced_flat = enhanced_result.flatten()
        correlation = torch.corrcoef(torch.stack([original_flat, enhanced_flat]))[0, 1].item()

        # Check mass conservation
        total_precip = torch.sum(torch.ones_like(dem) * precip * dx * dy).item()
        total_original = torch.sum(original_result).item()
        total_enhanced = torch.sum(enhanced_result).item()
        original_mass_error = (total_original - total_precip) / total_precip
        enhanced_mass_error = (total_enhanced - total_precip) / total_precip

        stats = {
            'max_abs_diff': max_abs_diff,
            'mean_abs_diff': mean_abs_diff,
            'max_rel_diff': max_rel_diff,
            'mean_rel_diff': mean_rel_diff,
            'correlation': correlation,
            'total_precip': total_precip,
            'total_original': total_original,
            'total_enhanced': total_enhanced,
            'original_mass_error': original_mass_error,
            'enhanced_mass_error': enhanced_mass_error
        }
    else:
        stats = None

    return {
        'dem': dem.cpu().numpy(),
        'original_result': original_result.cpu().numpy() if original_success else None,
        'enhanced_result': enhanced_result.cpu().numpy() if enhanced_success else None,
        'original_time': original_time,
        'enhanced_time': enhanced_time,
        'original_success': original_success,
        'enhanced_success': enhanced_success,
        'stats': stats
    }

def visualize_results(results):
    """
    Visualizes the comparison results.

    Args:
        results: Dictionary containing comparison results
    """
    dem = results['dem'][0, 0]
    original_result = results['original_result'][0, 0] if results['original_result'] is not None else None
    enhanced_result = results['enhanced_result'][0, 0] if results['enhanced_result'] is not None else None

    # Create figure
    fig = plt.figure(figsize=(15, 10))

    # Plot DEM
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(dem, cmap='terrain')
    ax1.set_title('DEM')
    plt.colorbar(im1, ax=ax1, label='Elevation')

    # Plot original result
    if original_result is not None:
        ax2 = fig.add_subplot(2, 3, 2)
        im2 = ax2.imshow(np.log1p(original_result), cmap='Blues')
        ax2.set_title('Original Drainage Area (log1p)')
        plt.colorbar(im2, ax=ax2, label='log(1 + Area)')

    # Plot enhanced result
    if enhanced_result is not None:
        ax3 = fig.add_subplot(2, 3, 3)
        im3 = ax3.imshow(np.log1p(enhanced_result), cmap='Blues')
        ax3.set_title('Enhanced Drainage Area (log1p)')
        plt.colorbar(im3, ax=ax3, label='log(1 + Area)')

    # Plot difference if both methods succeeded
    if original_result is not None and enhanced_result is not None:
        ax4 = fig.add_subplot(2, 3, 4)
        abs_diff = np.abs(enhanced_result - original_result)
        im4 = ax4.imshow(abs_diff, cmap='Reds')
        ax4.set_title('Absolute Difference')
        plt.colorbar(im4, ax=ax4, label='|Enhanced - Original|')

        ax5 = fig.add_subplot(2, 3, 5)
        rel_diff = abs_diff / (original_result + 1e-10)
        rel_diff = np.clip(rel_diff, 0, 1)  # Clip for better visualization
        im5 = ax5.imshow(rel_diff, cmap='Reds', vmax=0.5)
        ax5.set_title('Relative Difference (clipped to 0-1)')
        plt.colorbar(im5, ax=ax5, label='|Enhanced - Original| / Original')

        # Plot statistics
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        stats = results['stats']
        stats_text = (
            f"Statistics:\n"
            f"Original Time: {results['original_time']:.3f}s\n"
            f"Enhanced Time: {results['enhanced_time']:.3f}s\n"
            f"Speed Ratio: {results['original_time'] / results['enhanced_time']:.2f}x\n\n"
            f"Max Abs Diff: {stats['max_abs_diff']:.4f}\n"
            f"Mean Abs Diff: {stats['mean_abs_diff']:.4f}\n"
            f"Max Rel Diff: {stats['max_rel_diff']:.4f}\n"
            f"Mean Rel Diff: {stats['mean_rel_diff']:.4f}\n"
            f"Correlation: {stats['correlation']:.4f}\n\n"
            f"Total Precip: {stats['total_precip']:.4f}\n"
            f"Total Original: {stats['total_original']:.4f}\n"
            f"Total Enhanced: {stats['total_enhanced']:.4f}\n"
            f"Original Mass Error: {stats['original_mass_error']:.4%}\n"
            f"Enhanced Mass Error: {stats['enhanced_mass_error']:.4%}"
        )
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
                 fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.savefig('drainage_area_comparison.png', dpi=150)
    plt.show()

def main():
    """Main function to run the comparison."""
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create test DEM
    size = 128
    print(f"Creating test DEM of size {size}x{size}...")
    dem = create_test_dem(size=size, device=device)

    # Run comparison
    print("Running comparison...")
    results = run_comparison(dem)

    # Visualize results
    print("Visualizing results...")
    visualize_results(results)

    print("Done!")

if __name__ == "__main__":
    main()

import argparse
import logging
import os
import sys
import time
import torch
import numpy as np

# Add src directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.utils import setup_logging, get_device
from src.physics import calculate_drainage_area_differentiable_optimized
# Import other functions to benchmark as needed (e.g., interpolation, model forward)
# from src.losses import rbf_interpolate
# from src.models import FastscapePINN

def benchmark_drainage_area(grid_sizes, num_runs=5, device='cpu', da_params=None):
    """Benchmarks the optimized drainage area calculation."""
    if da_params is None:
        da_params = {'temp': 0.01, 'num_iters': 50} # Default params

    results = {}
    logging.info(f"--- Benchmarking Drainage Area (Device: {device}, Runs: {num_runs}) ---")
    logging.info(f"Drainage Area Params: {da_params}")

    for size in grid_sizes:
        H, W = size, size
        logging.info(f"Grid Size: {H}x{W}")
        # Create dummy topography data
        try:
            h = torch.rand(1, 1, H, W, device=device) * 100 # Example random topo
            dx, dy = 100.0, 100.0 # Example grid spacing
        except Exception as e:
            logging.error(f"Failed to create dummy data for size {size} on {device}: {e}. Skipping.")
            results[f"{H}x{W}"] = {'time_avg': float('nan'), 'time_std': float('nan')}
            continue

        timings = []
        # Warm-up run
        try:
            _ = calculate_drainage_area_differentiable_optimized(h, dx, dy, **da_params)
        except Exception as e:
             logging.error(f"Error during warm-up run for size {size}: {e}. Skipping benchmark.")
             results[f"{H}x{W}"] = {'time_avg': float('nan'), 'time_std': float('nan')}
             continue

        # Actual benchmark runs
        for _ in range(num_runs):
            start_time = time.perf_counter()
            try:
                _ = calculate_drainage_area_differentiable_optimized(h, dx, dy, **da_params)
                # Ensure CUDA operations finish if on GPU
                if device == 'cuda':
                    torch.cuda.synchronize()
            except Exception as e:
                 logging.error(f"Error during benchmark run for size {size}: {e}")
                 timings.append(float('nan'))
                 break # Stop runs for this size on error
            end_time = time.perf_counter()
            timings.append(end_time - start_time)

        if any(np.isnan(timings)):
             avg_time = float('nan')
             std_time = float('nan')
        else:
             avg_time = np.mean(timings)
             std_time = np.std(timings)

        results[f"{H}x{W}"] = {'time_avg': avg_time, 'time_std': std_time}
        logging.info(f"  Avg Time: {avg_time:.4f} +/- {std_time:.4f} seconds")
        # Clean up GPU memory if needed
        del h
        if device == 'cuda':
            torch.cuda.empty_cache()

    return results

# --- Add benchmark functions for other components (interpolation, model forward) here ---
# def benchmark_interpolation(...): ...
# def benchmark_model_forward(...): ...


def main(args):
    """Main function to run benchmarks."""
    # Setup logging for benchmark script
    log_dir = os.path.join(script_dir, 'logs_benchmark')
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir=log_dir)
    device = get_device({'device': args.device}) # Use util to get device

    grid_sizes = [int(s) for s in args.grid_sizes.split(',')]
    num_runs = args.num_runs

    # Benchmark Drainage Area
    da_results = benchmark_drainage_area(grid_sizes, num_runs, device)
    print("\n--- Drainage Area Benchmark Results ---")
    for size, res in da_results.items():
        print(f"Grid {size}: Avg Time = {res['time_avg']:.4f} +/- {res['time_std']:.4f} s")

    # --- Call other benchmark functions here ---
    # interp_results = benchmark_interpolation(...)
    # model_results = benchmark_model_forward(...)

    logging.info("Benchmarking finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run performance benchmarks for PINN_Fastscape_Framework components.")
    parser.add_argument('--device', type=str, default='auto', help="Device to run on ('auto', 'cpu', 'cuda').")
    parser.add_argument('--grid_sizes', type=str, default='32,64,128,256', help="Comma-separated list of grid sizes (N for NxN grids).")
    parser.add_argument('--num_runs', type=int, default=5, help="Number of runs for averaging timings.")
    # Add arguments for other benchmark parameters if needed

    args = parser.parse_args()
    main(args)
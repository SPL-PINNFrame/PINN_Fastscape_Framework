"""
Enhanced implementation of differentiable drainage area calculation.

This module provides an improved version of the drainage area calculation
with better numerical stability, convergence checking, and handling of
special cases like flat areas and depressions.
"""

import torch
import torch.nn.functional as F
import math
import logging
import warnings
from typing import Dict, Tuple, Union, Optional, List

# --- Helper Functions ---

def _shift_field(tensor: torch.Tensor, shift_y: int, shift_x: int) -> torch.Tensor:
    """
    Shifts a tensor (B, C, H, W) by (shift_y, shift_x) using padding and slicing.
    Pads with zeros for areas shifted in from outside the domain.

    Args:
        tensor: Input tensor of shape (B, C, H, W)
        shift_y: Vertical shift (positive = down)
        shift_x: Horizontal shift (positive = right)

    Returns:
        Shifted tensor of same shape as input
    """
    B, C, H, W = tensor.shape
    # Calculate padding: (pad_left, pad_right, pad_top, pad_bottom)
    pad_y = (max(0, -shift_y), max(0, shift_y))  # Pad top, bottom
    pad_x = (max(0, -shift_x), max(0, shift_x))  # Pad left, right

    # Use F.pad with 'constant' mode and value=0
    padded = F.pad(tensor, pad=(pad_x[0], pad_x[1], pad_y[0], pad_y[1]), mode='constant', value=0)

    # Crop back to original size after shift
    start_y = max(0, shift_y)
    end_y = start_y + H
    start_x = max(0, shift_x)
    end_x = start_x + W

    return padded[:, :, start_y:end_y, start_x:end_x]

def _calculate_gradients(h: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates elevation gradient (dy, dx) and normalized gradient vector.

    Args:
        h: Elevation tensor of shape (B, 1, H, W)
        eps: Small epsilon for numerical stability

    Returns:
        Tuple of (gradient tensor, normalized gradient tensor)
    """
    device = h.device
    dtype = h.dtype

    # Sobel kernels for gradient calculation
    sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                  dtype=dtype, device=device).view(1, 1, 3, 3)
    sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                  dtype=dtype, device=device).view(1, 1, 3, 3)

    # Calculate gradients using convolution (padding=1 keeps size H,W)
    grad_x = F.conv2d(h, sobel_x_kernel, padding=1)  # Gradient along x (right positive)
    grad_y = F.conv2d(h, sobel_y_kernel, padding=1)  # Gradient along y (down positive)

    # Stack gradients: order (dy, dx) consistent with shifts array later
    grad = torch.cat([grad_y, grad_x], dim=1)  # Shape (B, 2, H, W)

    # Normalize gradient vector (points in direction of steepest *ascent*)
    # Add eps for stability when gradient magnitude is near zero
    grad_norm = F.normalize(grad, p=2, dim=1, eps=eps)  # Shape (B, 2, H, W)

    return grad, grad_norm

def _calculate_flow_weights(
    h_center: torch.Tensor,
    h_neighbors: torch.Tensor,
    grad_norm: torch.Tensor,
    neighbor_distances: torch.Tensor,
    neighbor_shifts_normalized: torch.Tensor,
    current_temp: float,
    lambda_dir: float,
    eps: float,
    flat_handling: str = 'uniform'  # 'uniform', 'gradient', or 'none'
) -> torch.Tensor:
    """
    Calculates the flow weights to the 8 neighbors based on slope and gradient direction.

    Args:
        h_center: Center elevation tensor of shape (B, 1, H, W)
        h_neighbors: Neighbor elevations tensor of shape (B, 8, H, W)
        grad_norm: Normalized gradient tensor of shape (B, 2, H, W)
        neighbor_distances: Distances to neighbors tensor of shape (1, 8, 1, 1)
        neighbor_shifts_normalized: Normalized neighbor offset vectors of shape (8, 2)
        current_temp: Current temperature parameter for Softmax
        lambda_dir: Weight for gradient direction constraint
        eps: Small epsilon for numerical stability

    Returns:
        Flow weights tensor of shape (B, 8, H, W)
    """
    # Calculate height difference: dh = h_center - h_neighbor
    # Positive dh means downhill in that direction
    dh = h_center - h_neighbors  # Shape (B, 8, H, W)

    # --- Gradient Direction Constraint ---
    # Reshape for calculating cosine similarity
    # grad_norm: (B, 2, H, W) -> (B, H, W, 2) -> (B, H, W, 1, 2)
    grad_norm_exp = grad_norm.permute(0, 2, 3, 1).unsqueeze(3)
    # neighbor_shifts_normalized: (8, 2) -> (1, 1, 1, 8, 2)
    neighbor_shifts_norm_exp = neighbor_shifts_normalized.view(1, 1, 1, 8, 2)

    # Cosine similarity between negative gradient (-grad_norm, steepest descent) and neighbor vectors.
    # Result shape: (B, H, W, 8)
    cos_similarity = (-grad_norm_exp * neighbor_shifts_norm_exp).sum(dim=-1)
    # Permute to match dh shape: (B, 8, H, W)
    cos_similarity = cos_similarity.permute(0, 3, 1, 2)

    # --- Calculate Logits ---
    # Slope term: positive for downhill, scaled by distance and temperature
    # Note: Using dh/distance is closer to actual slope definition
    slope_term = (dh / neighbor_distances.clamp(min=eps)) / current_temp  # Clamp distance just in case

    # Direction term
    direction_term = lambda_dir * cos_similarity

    # Combine terms - Additive combination
    logits = slope_term + direction_term

    # --- Masking and Stability ---
    # Mask: Only allow flow downhill (dh > eps to handle near-zero slopes)
    mask_downhill = (dh > eps).float()

    # 检测平坦区域（所有邻居高度差都很小）
    is_flat = (torch.sum(mask_downhill, dim=1, keepdim=True) < 0.5)

    # 根据flat_handling参数处理平坦区域
    if flat_handling == 'uniform':
        # 对平坦区域使用均匀分布（所有方向相等权重）
        uniform_logits = torch.zeros_like(logits)
        # 应用掩码：非平坦区域使用正常logits，平坦区域使用uniform_logits
        logits = torch.where(is_flat, uniform_logits, logits)
        # 对非平坦区域应用下坡掩码
        non_flat_mask = (~is_flat).float()
        logits = logits * mask_downhill * non_flat_mask + \
                (1 - mask_downhill * non_flat_mask) * -1e10  # 大的负数
    elif flat_handling == 'gradient':
        # 对平坦区域使用梯度方向（即使高度差很小）
        flat_logits = lambda_dir * cos_similarity * 10.0  # 增强梯度影响
        # 应用掩码：非平坦区域使用正常logits，平坦区域使用flat_logits
        logits = torch.where(is_flat, flat_logits, logits)
        # 对非平坦区域应用下坡掩码
        non_flat_mask = (~is_flat).float()
        logits = logits * (mask_downhill * non_flat_mask + is_flat.float()) + \
                (1 - (mask_downhill * non_flat_mask + is_flat.float())) * -1e10
    else:  # 'none' - 使用原始方法
        # Apply mask using a large negative number for non-downhill/flat directions
        # This effectively sets their probability close to zero after softmax
        logits = logits * mask_downhill + (1 - mask_downhill) * -1e10  # Large negative logit

    # Clamp logits for numerical stability before softmax
    # Prevents potential overflow/underflow issues with exp() in softmax
    logits = torch.clamp(logits, min=-60.0, max=60.0)

    # Calculate weights using softmax - represents fraction of flow going to each neighbor
    # dim=1 corresponds to the 8 neighbors dimension
    weights = torch.softmax(logits, dim=1)  # Shape: (B, 8, H, W)

    # Handle potential NaNs from softmax (e.g., if all logits were -inf due to a pit)
    # Replace NaN with 0.0, ensuring outflow from pits is zero.
    weights = torch.nan_to_num(weights, nan=0.0)

    return weights

def _handle_depressions(h: torch.Tensor, drainage_area: torch.Tensor, local_flow: torch.Tensor) -> torch.Tensor:
    """
    Special handling for depressions (pits) in the topography.
    Identifies local minima and enhances their drainage area.

    Args:
        h: Elevation tensor of shape (B, 1, H, W)
        drainage_area: Current drainage area tensor of shape (B, 1, H, W)
        local_flow: Local flow tensor of shape (B, 1, H, W)

    Returns:
        Updated drainage area tensor of shape (B, 1, H, W)
    """
    batch_size, _, height, width = h.shape

    # Find local minima (pits)
    h_pad = F.pad(h, (1, 1, 1, 1), mode='constant', value=float('inf'))
    is_min = torch.ones_like(h, dtype=torch.bool)

    # Check all 8 neighbors to identify local minima
    offsets = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1)
    ]

    for dy_offset, dx_offset in offsets:
        neighbor_h = h_pad[:, :, 1+dy_offset:height+1+dy_offset, 1+dx_offset:width+1+dx_offset]
        is_min = is_min & (h <= neighbor_h)

    # Enhance drainage area at local minima
    # The enhancement factor can be adjusted based on the application
    enhancement_factor = 5.0
    min_mask = is_min.float() * enhancement_factor

    # Apply enhancement
    enhanced_drainage_area = drainage_area * (1.0 + min_mask)

    return enhanced_drainage_area

def _check_mass_conservation(drainage_area: torch.Tensor, local_flow: torch.Tensor, iteration: int) -> float:
    """
    Checks mass conservation by comparing total drainage area to total local flow.

    Args:
        drainage_area: Current drainage area tensor
        local_flow: Local flow tensor (precipitation * cell area)
        iteration: Current iteration number (for logging)

    Returns:
        Relative difference between total drainage area and total local flow
    """
    # Use torch.sum with keepdim=True to avoid scalar tensor issues
    total_drainage = torch.sum(drainage_area).item()
    total_local_flow = torch.sum(local_flow).item()

    # Check for valid values
    if not (math.isfinite(total_drainage) and math.isfinite(total_local_flow)):
        logging.warning(f"Iteration {iteration}: Non-finite values detected in mass conservation check")
        return float('nan')

    # Calculate relative difference with safety checks
    if abs(total_local_flow) < 1e-10:
        relative_diff = 0.0 if abs(total_drainage) < 1e-10 else float('inf')
    else:
        relative_diff = (total_drainage - total_local_flow) / total_local_flow

    # Log at debug level for normal operation
    logging.debug(f"Iteration {iteration}: Total drainage area = {total_drainage:.4f}, "
                 f"Total local flow = {total_local_flow:.4f}, "
                 f"Relative difference = {relative_diff:.6f}")

    return relative_diff

# --- Main Function ---

def calculate_drainage_area_enhanced(
    h: torch.Tensor,
    dx: float,
    dy: float,
    precip: Union[float, torch.Tensor] = 1.0,
    initial_temp: float = 0.01,
    end_temp: float = 1e-5,
    annealing_factor: float = 0.98,
    max_iters: int = 50,  # Reduced default max iterations
    lambda_dir: float = 1.0,
    convergence_threshold: float = 1e-5,
    eps: float = 1e-6,
    special_depression_handling: bool = True,
    check_mass_conservation: bool = False,
    stable_mode: bool = False,  # New parameter for extra stability
    flat_handling: str = 'uniform',  # 'uniform', 'gradient', or 'none'
    clamp_max_value: float = 1e6,  # Maximum value for clamping
    verbose: bool = False
) -> torch.Tensor:
    """
    Enhanced differentiable drainage area calculation with improved stability,
    convergence checking, and special handling for depressions.

    Args:
        h: Topography elevation tensor of shape (B, 1, H, W)
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction
        precip: Precipitation rate (scalar or tensor of shape (B, 1, H, W))
        initial_temp: Initial temperature for Softmax
        end_temp: Minimum temperature for annealing
        annealing_factor: Factor to decrease temperature each iteration
        max_iters: Maximum number of iterations
        lambda_dir: Weight for gradient direction constraint
        convergence_threshold: Threshold for maximum change to determine convergence
        eps: Small epsilon for numerical stability
        special_depression_handling: Whether to apply special handling for depressions
        check_mass_conservation: Whether to check mass conservation during iterations
        verbose: Whether to print progress information

    Returns:
        Calculated drainage area tensor of shape (B, 1, H, W)
    """
    # --- 1. Input Validation ---
    if not isinstance(h, torch.Tensor) or h.ndim != 4:
        raise ValueError(f"Input heightmap 'h' must be a 4D tensor (B, 1, H, W), got ndim={h.ndim}")
    if h.shape[1] != 1:
        raise ValueError(f"Input heightmap 'h' must have 1 channel, got shape={h.shape}")
    if dx <= 0 or dy <= 0:
        raise ValueError(f"Grid spacings 'dx' ({dx}) and 'dy' ({dy}) must be positive.")
    if initial_temp <= 0 or end_temp <= 0 or annealing_factor <= 0 or annealing_factor >= 1:
        raise ValueError("Temperature parameters (initial, end, factor) must be positive, and factor < 1.")

    batch_size, _, height, width = h.shape
    device = h.device
    dtype = h.dtype
    cell_area = dx * dy

    if verbose:
        logging.info(f"Starting enhanced drainage area calculation:")
        logging.info(f"  Input shape: {h.shape}, Device: {device}, Cell area: {cell_area}")
        logging.info(f"  Parameters: initial_temp={initial_temp:.2e}, end_temp={end_temp:.2e}, "
                    f"annealing_factor={annealing_factor}, max_iters={max_iters}")

    # --- 2. Initialize Local Flow ---
    if isinstance(precip, (int, float)):
        local_flow = torch.full_like(h, float(precip) * cell_area)
    elif isinstance(precip, torch.Tensor):
        if precip.shape != h.shape:
            raise ValueError(f"Precipitation tensor shape {precip.shape} must match heightmap shape {h.shape}")
        local_flow = precip * cell_area
    else:
        raise TypeError(f"Unsupported type for 'precip': {type(precip)}")

    # --- 3. Precompute Neighbor Information ---
    # Define 8 neighbors offsets (y, x) relative to center: [N, NE, E, SE, S, SW, W, NW]
    shifts = torch.tensor([
        [-1, 0], [-1, 1], [0, 1], [1, 1],
        [1, 0], [1, -1], [0, -1], [-1, -1]
    ], dtype=torch.long, device=device)

    # Reverse shifts (from neighbor back to center)
    reverse_shifts = -shifts

    # Calculate distances to neighbors
    distances = torch.sqrt((shifts[:, 0] * dy)**2 + (shifts[:, 1] * dx)**2).to(dtype)
    distances_view = distances.view(1, 8, 1, 1)  # Shape (1, 8, 1, 1) for broadcasting

    # Normalize neighbor offset vectors for direction constraint
    shifts_normalized = F.normalize(shifts.to(dtype), p=2, dim=1, eps=eps)  # Shape (8, 2)

    # --- 4. Precompute Neighbor Heights ---
    # Pad elevation map for neighbor access
    padded_h = F.pad(h, pad=(1, 1, 1, 1), mode='reflect')
    # Extract 3x3 patches efficiently
    unfolded_h = F.unfold(padded_h, kernel_size=3).view(batch_size, 9, height, width)
    h_center = unfolded_h[:, 4:5, :, :]  # Center element index is 4

    # Indices of the 8 neighbors in the 3x3 patch (order matches `shifts`)
    neighbor_indices = [1, 2, 5, 8, 7, 6, 3, 0]
    h_neighbors = unfolded_h[:, neighbor_indices, :, :]  # Shape (B, 8, H, W)

    # --- 5. Calculate Gradients (once, since DEM is static) ---
    grad, grad_norm = _calculate_gradients(h, eps)

    # Use stable mode if requested (more conservative approach)
    if stable_mode:
        # Use higher initial temperature
        initial_temp = max(initial_temp, 0.1)
        # Use slower annealing
        annealing_factor = max(annealing_factor, 0.99)
        # Limit max iterations
        max_iters = min(max_iters, 20)
        # Use higher convergence threshold
        if convergence_threshold > 0:
            convergence_threshold = max(convergence_threshold, 1e-3)

    # --- 6. Iterative Flow Accumulation ---
    drainage_area = local_flow.clone()
    previous_drainage_area = torch.zeros_like(drainage_area)
    temp = initial_temp
    converged = False

    if check_mass_conservation:
        total_initial_flow = torch.sum(local_flow).item()
        logging.info(f"Total initial flow (precip * area): {total_initial_flow:.4f}")

    for it in range(max_iters):
        # Store previous state for convergence check
        previous_drainage_area = drainage_area.clone()

        # Calculate flow weights with current temperature
        current_temp = max(temp, end_temp)  # Apply floor to temperature
        weights = _calculate_flow_weights(
            h_center, h_neighbors, grad_norm, distances_view, shifts_normalized,
            current_temp, lambda_dir, eps, flat_handling
        )

        # Calculate outflow from each cell to its neighbors
        # Clamp drainage_area to prevent extreme values
        # Use a much lower max value for better stability
        safe_drainage_area = torch.clamp(drainage_area, min=0.0, max=clamp_max_value)
        outflow_to_neighbors = safe_drainage_area * weights  # Shape (B, 8, H, W)

        # Calculate inflow by summing contributions shifted from neighbors
        inflow = torch.zeros_like(drainage_area)
        for k in range(8):
            # Get the outflow calculated *at* the neighbor k location
            outflow_from_neighbor_k = outflow_to_neighbors[:, k:k+1, :, :]
            # Shift this outflow from neighbor k's location *to* the current cell's location
            inflow += _shift_field(outflow_from_neighbor_k,
                                  reverse_shifts[k, 0].item(),
                                  reverse_shifts[k, 1].item())

        # Update drainage area: A_new = Precip + Inflow
        # Clamp inflow to prevent numerical issues
        # Use a much lower max value for better stability
        safe_inflow = torch.clamp(inflow, min=0.0, max=clamp_max_value)
        drainage_area = local_flow + safe_inflow

        # Apply global clamp to drainage_area as a safety measure
        drainage_area = torch.clamp(drainage_area, min=0.0, max=clamp_max_value)

        # Special handling for depressions if enabled
        if special_depression_handling:
            # Only apply depression handling after a few iterations
            if it >= 5:  # Allow some normal flow accumulation first
                drainage_area = _handle_depressions(h, drainage_area, local_flow)

        # Check for NaN/Inf values
        if torch.isnan(drainage_area).any() or torch.isinf(drainage_area).any():
            warnings.warn(f"NaN or Inf detected in drainage_area at iteration {it+1}. "
                         "Replacing with previous valid values.")
            # Replace with previous valid values
            drainage_area = torch.where(
                torch.isnan(drainage_area) | torch.isinf(drainage_area),
                previous_drainage_area,
                drainage_area
            )
            # If previous was also invalid, use local_flow as fallback
            drainage_area = torch.where(
                torch.isnan(drainage_area) | torch.isinf(drainage_area),
                local_flow,
                drainage_area
            )

        # Check mass conservation if enabled
        if check_mass_conservation and (it % 10 == 0 or it == max_iters - 1):
            rel_diff = _check_mass_conservation(drainage_area, local_flow, it)
            if math.isfinite(rel_diff) and abs(rel_diff) > 0.01:  # More than 1% difference
                logging.warning(f"Iteration {it}: Mass conservation error: {rel_diff:.2%}")

        # Check convergence
        if convergence_threshold > 0:
            max_change = torch.max(torch.abs(drainage_area - previous_drainage_area)).item()
            if max_change < convergence_threshold:
                converged = True
                if verbose:
                    logging.info(f"Converged at iteration {it+1} with max change {max_change:.2e}")
                break

        # Anneal temperature
        temp = temp * annealing_factor

        # Log progress
        if verbose and (it % 10 == 0 or it == max_iters - 1):
            logging.info(f"Iteration {it+1}/{max_iters}, Temp: {current_temp:.2e}, "
                        f"Max DA: {torch.max(drainage_area).item():.2f}")

    # Log convergence status
    if not converged and convergence_threshold > 0:
        warnings.warn(f"Drainage area calculation did not converge within {max_iters} iterations. "
                     f"Final max change: {max_change:.2e}")

    # Final safety check for NaNs
    drainage_area = torch.nan_to_num(drainage_area, nan=0.0)

    return drainage_area

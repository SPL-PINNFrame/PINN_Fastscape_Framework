import torch
import torch.nn.functional as F
import numpy as np # Keep for isnan check in compute_total_loss
import logging # Add logging
# import numpy as np # No longer needed for griddata
# from scipy.interpolate import griddata # Replaced with RBF interpolation
# Import necessary functions from the updated physics module
from .physics import calculate_dhdt_physics, calculate_slope_magnitude

# --- Data Fidelity Loss ---

def compute_data_loss(predicted_topo, target_topo):
    """
    Computes the data fidelity loss (e.g., MSE) between predicted and target topography.
    Assumes predicted_topo and target_topo correspond to the same time instance(s)
    and have the same grid shape [B, C, H, W].
    """
    if predicted_topo.shape != target_topo.shape:
         logging.warning(f"Shape mismatch in compute_data_loss: pred={predicted_topo.shape}, target={target_topo.shape}. Resizing target.")
         # Example: Resize target to match prediction using interpolation
         target_topo = F.interpolate(target_topo.float(), size=predicted_topo.shape[-2:], mode='bilinear', align_corners=False)

    # Ensure target_topo is also float for mse_loss
    return F.mse_loss(predicted_topo, target_topo.float())

# --- PDE Residual Calculation (Original - Interpolation Based) ---

# Helper function for RBF interpolation
def rbf_interpolate(values, points, query_points, sigma=0.1, device='cpu'):
    """Radial Basis Function (RBF) differentiable interpolation using PyTorch."""
    # Ensure inputs are torch tensors on the correct device
    values = values.to(device)
    points = points.to(device)
    query_points = query_points.to(device)

    # Calculate point-to-point distance matrix squared
    diff = query_points.unsqueeze(1) - points.unsqueeze(0)  # [Q, P, 2]
    dist_sq = torch.sum(diff**2, dim=2)  # [Q, P]

    # Calculate RBF weights (Gaussian kernel)
    weights = torch.exp(-dist_sq / (2 * sigma**2))  # [Q, P]

    # Normalize weights to sum to 1 for each query point
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-10) # Add epsilon for stability

    # Weighted average to compute interpolated values
    # Ensure values has shape [P, C] where C is number of channels/features
    if values.ndim == 1:
        values = values.unsqueeze(1) # Add channel dim if needed: [P] -> [P, 1]

    interp_values = torch.matmul(weights, values)  # [Q, P] @ [P, C] -> [Q, C]
    return interp_values


def compute_pde_residual(h_pred, coords, physics_params):
    """
    Computes the residual of the governing physical PDE using RBF interpolation.
    Residual = dh/dt - (U - K_f*A^m*S^n + K_d*Laplacian(h))

    Args:
        h_pred (torch.Tensor): Topography predicted by the PINN at collocation points.
                               Shape: (N_collocation, 1) or similar. Requires grad wrt time.
        coords (dict): Dictionary containing tensors for 'x', 'y', 't' coordinates
                       of the collocation points. Each tensor shape: (N_collocation, 1).
        physics_params (dict): Dictionary containing physical parameters like
                               U (uplift), K_f, m, n, K_d, dx, dy, etc.

    Returns:
        torch.Tensor: Mean squared residual of the PDE.
    """
    t_coords = coords['t']
    # Ensure h_pred requires gradients with respect to time coordinate t
    if not t_coords.requires_grad:
         t_coords.requires_grad_(True)
    if not h_pred.requires_grad:
         h_pred.requires_grad_(True)


    # 1. Calculate dh/dt using automatic differentiation
    try:
        dh_dt_pred = torch.autograd.grad(
            outputs=h_pred,
            inputs=t_coords,
            grad_outputs=torch.ones_like(h_pred),
            create_graph=True, # Keep graph for potential higher-order derivatives or backprop through loss
            retain_graph=True, # Keep graph as physics terms also depend on h_pred
            allow_unused=False # Ensure t_coords is actually used in h_pred computation
        )[0]
    except RuntimeError as e:
        logging.error(f"Error computing dh/dt in compute_pde_residual: {e}. Check if t_coords influences h_pred.")
        return h_pred.sum() * 0.0 # Return zero loss with grad connection

    # 2. Interpolate scattered predictions (h_pred, dh_dt_pred) onto a regular grid
    # Get grid parameters from physics_params
    grid_height = physics_params.get('grid_height')
    grid_width = physics_params.get('grid_width')
    domain_x = physics_params.get('domain_x', [0.0, physics_params.get('dx', 1.0) * (grid_width - 1)]) # Estimate if not provided
    domain_y = physics_params.get('domain_y', [0.0, physics_params.get('dy', 1.0) * (grid_height - 1)]) # Estimate if not provided
    dx = physics_params.get('dx', (domain_x[1] - domain_x[0]) / (grid_width - 1) if grid_width > 1 else 1.0)
    dy = physics_params.get('dy', (domain_y[1] - domain_y[0]) / (grid_height - 1) if grid_height > 1 else 1.0)

    if grid_height is None or grid_width is None:
         logging.warning("Grid dimensions (grid_height, grid_width) not found in physics_params. Skipping PDE residual calculation (interpolation method).")
         return h_pred.sum() * 0.0

    # Create target grid coordinates normalized to [0, 1] for RBF interpolation
    grid_x_norm = torch.linspace(0, 1, grid_width, device=h_pred.device)
    grid_y_norm = torch.linspace(0, 1, grid_height, device=h_pred.device)
    grid_y_mesh_norm, grid_x_mesh_norm = torch.meshgrid(grid_y_norm, grid_x_norm, indexing='ij') # H, W
    grid_points_norm = torch.stack([grid_x_mesh_norm.flatten(), grid_y_mesh_norm.flatten()], dim=1) # (H*W, 2)

    # Normalize collocation point coordinates to [0, 1]
    x_coords = coords['x'] # Shape (N, 1)
    y_coords = coords['y'] # Shape (N, 1)
    x_coords_norm = (x_coords - domain_x[0]) / (domain_x[1] - domain_x[0] + 1e-9) # Add epsilon for zero range
    y_coords_norm = (y_coords - domain_y[0]) / (domain_y[1] - domain_y[0] + 1e-9)
    scattered_points_norm = torch.cat([x_coords_norm, y_coords_norm], dim=1) # Shape (N, 2)

    # Interpolate h_pred and dh_dt_pred onto the normalized grid using RBF
    rbf_sigma = physics_params.get('rbf_sigma', 0.1)
    try:
        h_pred_interp = rbf_interpolate(h_pred, scattered_points_norm, grid_points_norm, sigma=rbf_sigma, device=h_pred.device) # Shape (H*W, 1)
        dh_dt_pred_interp = rbf_interpolate(dh_dt_pred, scattered_points_norm, grid_points_norm, sigma=rbf_sigma, device=h_pred.device) # Shape (H*W, 1)
    except Exception as e:
         logging.error(f"Error during RBF interpolation: {e}. Skipping PDE residual calculation.", exc_info=True)
         logging.error(f"Shapes - h_pred: {h_pred.shape}, points: {scattered_points_norm.shape}, query: {grid_points_norm.shape}")
         return h_pred.sum() * 0.0


    # Reshape interpolated values back to grid (B=1, C=1, H, W)
    try:
        h_pred_grid = h_pred_interp.reshape(1, 1, grid_height, grid_width)
        dh_dt_pred_grid = dh_dt_pred_interp.reshape(1, 1, grid_height, grid_width)
    except RuntimeError as e:
        logging.error(f"Error reshaping interpolated values: {e}. H={grid_height}, W={grid_width}, NumPoints={h_pred_interp.numel()}")
        return h_pred.sum() * 0.0

    # 3. Calculate the physics-based tendency (RHS of the PDE) using the interpolated grid
    # Extract necessary parameters
    U = physics_params.get('U', 0.0)
    K_f = physics_params.get('K_f', 1e-5)
    m = physics_params.get('m', 0.5)
    n = physics_params.get('n', 1.0)
    K_d = physics_params.get('K_d', 0.01)
    precip = physics_params.get('precip', 1.0)
    da_kwargs = physics_params.get('drainage_area_kwargs', {})

    try:
        dhdt_physics = calculate_dhdt_physics(
            h=h_pred_grid, # Use the interpolated grid
            U=U, K_f=K_f, m=m, n=n, K_d=K_d, dx=dx, dy=dy,
            precip=precip,
            da_optimize_params=da_kwargs # Pass drainage area specific params
        )
    except Exception as e:
         logging.error(f"Error during calculate_dhdt_physics (interpolation method): {e}. Skipping PDE residual calculation.", exc_info=True)
         return h_pred_grid.sum() * 0.0


    # 4. Calculate the residual on the grid
    pde_residual = dh_dt_pred_grid - dhdt_physics

    # Return the mean squared residual
    return F.mse_loss(pde_residual, torch.zeros_like(pde_residual))


# --- PDE Residual Calculation (Adaptive - Local Point Based) ---

def sample_from_grid(param_grid, x_coords, y_coords):
    """在参数网格上采样局部值 (基于模型中的 _sample_at_coords)

    Args:
        param_grid (torch.Tensor or None): 参数场 (e.g., K or U), shape [B, 1, H, W] or [H, W].
        x_coords (torch.Tensor): 归一化 x 坐标 [0, 1], shape [N, 1].
        y_coords (torch.Tensor): 归一化 y 坐标 [0, 1], shape [N, 1].

    Returns:
        torch.Tensor: 在坐标点采样得到的值, shape [N, 1].
    """
    if param_grid is None:
        return torch.zeros_like(x_coords) # Return zeros if no grid provided

    device = x_coords.device
    param_grid = param_grid.to(device)

    # Ensure param_grid has batch and channel dimensions [B, C, H, W]
    if param_grid.ndim == 2: # [H, W]
        param_grid = param_grid.unsqueeze(0).unsqueeze(0)
    elif param_grid.ndim == 3: # [B, H, W]
        param_grid = param_grid.unsqueeze(1)
    elif param_grid.ndim != 4:
        raise ValueError(f"Unsupported param_grid dimension: {param_grid.ndim}. Expected 2, 3, or 4.")

    # Clamp normalized coordinates to ensure they are within [0, 1]
    x_norm = torch.clamp(x_coords, 0, 1)
    y_norm = torch.clamp(y_coords, 0, 1)

    # Convert normalized coordinates [0, 1] to grid_sample coordinates [-1, 1]
    x_sample = 2.0 * x_norm - 1.0
    y_sample = 2.0 * y_norm - 1.0

    # Prepare sampling grid for F.grid_sample: shape [B, N, 1, 2] or [B, 1, N, 2]
    # grid_sample expects grid in (x, y) order
    grid = torch.stack([x_sample, y_sample], dim=-1) # Shape [N, 1, 2]
    grid = grid.unsqueeze(0) # Shape [1, N, 1, 2]
    # Expand grid to match batch size of param_grid if necessary
    if grid.shape[0] != param_grid.shape[0]:
         grid = grid.expand(param_grid.shape[0], -1, -1, -1)

    # Use grid_sample for differentiable sampling
    # align_corners=False is generally recommended
    sampled_values = F.grid_sample(param_grid.float(), grid.float(), mode='bilinear', padding_mode='border', align_corners=False) # Ensure float type
    # Output shape: [B, C, N, 1]

    # Reshape to [N, C] or [N, 1] if C=1
    # Average over batch dimension if B > 1 (or handle differently if needed)
    if sampled_values.shape[0] > 1:
         sampled_values = sampled_values.mean(dim=0)
    else:
         sampled_values = sampled_values.squeeze(0)

    # sampled_values shape is now [C, N, 1]
    # We want [N, C]
    return sampled_values.squeeze(-1).permute(1, 0) # [C, N] -> [N, C]

# Removed compute_local_physics function (lines ~226-339)
# Removed compute_pde_residual_adaptive function (lines ~341-414)


# --- NEW: PDE Residual Calculation (Grid Focused) ---

def compute_grid_temporal_derivative(h_grid, t_grid):
    """计算网格状态相对于时间的导数，保留网格结构

    Args:
        h_grid (torch.Tensor): Topography grid [B, C, H, W]. Requires grad.
        t_grid (torch.Tensor): Time tensor associated with h_grid.
                               Can be scalar, [B], [B, 1, 1, 1], or [B, C, H, W].
                               Must require grad.

    Returns:
        torch.Tensor: Temporal derivative dh/dt, shape [B, C, H, W].
    """
    if not t_grid.requires_grad:
        logging.warning("t_grid does not require grad in compute_grid_temporal_derivative. Enabling.")
        t_grid.requires_grad_(True)
    if not h_grid.requires_grad:
        # This might indicate an issue upstream if h_grid is detached
        logging.warning("h_grid does not require grad in compute_grid_temporal_derivative. Enabling.")
        h_grid.requires_grad_(True)

    # Use broadcasting for grad_outputs
    ones_like_output = torch.ones_like(h_grid)

    try:
        # Calculate gradient. The output shape should match h_grid.
        grads = torch.autograd.grad(
            outputs=h_grid,
            inputs=t_grid,
            grad_outputs=ones_like_output,
            create_graph=True,
            retain_graph=True, 
            allow_unused=False  # 修改: 不再允许 t_grid 未使用
        )
        
        # 如果代码执行到这里，说明 t_grid 确实影响了 h_grid (否则会引发异常)
        dh_dt = grads[0]
        
        # 确保输出形状与 h_grid 匹配（如果 t_grid 是标量/广播）
        if dh_dt.shape != h_grid.shape:
            logging.warning(f"形状不匹配: dh_dt={dh_dt.shape}, h_grid={h_grid.shape}. 尝试调整形状。")
            
            # 尝试处理特定形状的梯度
            if dh_dt.numel() == 1:  # 标量梯度
                dh_dt = dh_dt.expand_as(h_grid)
            elif dh_dt.shape == (h_grid.shape[0], 1, 1, 1):  # [B, 1, 1, 1] 形状
                dh_dt = dh_dt.expand_as(h_grid)
            else:
                # 如果无法解决形状不匹配，记录错误并返回备选输出
                logging.error(f"无法解决时间导数的形状不匹配 (dh_dt: {dh_dt.shape}, h_grid: {h_grid.shape})。")
                # 返回连接了梯度的零张量
                return h_grid.new_zeros(h_grid.shape, requires_grad=True)
        
        return dh_dt

    except RuntimeError as e:
        logging.error(f"计算网格时间导数时出错: {e}. t_grid 可能未影响 h_grid。", exc_info=True)
        logging.warning("无法计算时间导数。请改用 dual_output 模式，该模式由模型直接预测导数。")
        
        # 返回连接了梯度的零张量，但警告用户应该使用 dual_output 模式
        return h_grid.new_zeros(h_grid.shape, requires_grad=True)


def compute_pde_residual_grid_focused(h_pred_grid, t_grid, physics_params):
    """专注于网格的PDE残差计算

    Args:
        h_pred_grid (torch.Tensor): Predicted topography grid [B, C, H, W]. Requires grad.
        t_grid (torch.Tensor): Time tensor associated with h_pred_grid. Requires grad.
                               Shape should be compatible for autograd (e.g., scalar, [B], [B,1,1,1]).
        physics_params (dict): Dictionary containing physical parameters.

    Returns:
        torch.Tensor: Mean squared residual of the PDE over the grid.
    """
    # 1. 确保输入为网格表示
    if h_pred_grid.ndim != 4:
        logging.error(f"PDE residual calculation requires input grid format [B, C, H, W], got {h_pred_grid.shape}")
        # Return zero loss with grad connection
        return h_pred_grid.sum() * 0.0

    # 2. 计算时间导数（网格整体）
    dh_dt_grid = compute_grid_temporal_derivative(h_pred_grid, t_grid)
    if dh_dt_grid is None: # Handle potential error from derivative calculation
         return h_pred_grid.sum() * 0.0

    # 3. 直接在网格上计算物理项
    try:
        # Extract necessary parameters
        U = physics_params.get('U', 0.0)
        K_f = physics_params.get('K_f', 1e-5)
        m = physics_params.get('m', 0.5)
        n = physics_params.get('n', 1.0)
        K_d = physics_params.get('K_d', 0.01)
        dx = physics_params.get('dx', 1.0)
        dy = physics_params.get('dy', 1.0)
        precip = physics_params.get('precip', 1.0)
        da_kwargs = physics_params.get('drainage_area_kwargs', {})
        # Explicitly extract K_f, m, n, K_d from physics_params and ensure they are floats
        K_f = float(physics_params.get('K_f', 1e-5)) # Add default value
        m = float(physics_params.get('m', 0.5))     # Add default value
        n = float(physics_params.get('n', 1.0))     # Add default value
        K_d = float(physics_params.get('K_d', 0.01))   # Add default value


        # Handle potential spatial variation in parameters (U, K_f, K_d)
        # If they are tensors, ensure they match the grid shape or can be broadcast
        def prepare_param(param_val, target_shape, device):
            if isinstance(param_val, torch.Tensor):
                param_val = param_val.to(device)
                if param_val.shape == target_shape:
                    return param_val
                elif param_val.numel() == 1: # Scalar tensor
                    return param_val.expand(target_shape)
                elif param_val.ndim == 1 and param_val.shape[0] == target_shape[0]: # Batch dimension
                    return param_val.view(-1, 1, 1, 1).expand(target_shape)
                else:
                    # Attempt broadcasting for other shapes (e.g., spatial field without batch)
                    try:
                        return param_val.expand(target_shape)
                    except RuntimeError:
                         logging.error(f"Cannot broadcast parameter shape {param_val.shape} to target {target_shape}")
                         raise ValueError(f"Parameter shape mismatch: {param_val.shape} vs {target_shape}")
            else: # Scalar float/int
                return torch.full(target_shape, float(param_val), device=device)

        device = h_pred_grid.device
        target_shape = h_pred_grid.shape
        U_grid = prepare_param(U, target_shape, device)
        # K_f, m, n, K_d are typically scalar, but handle tensor case if needed
        # K_f_grid = prepare_param(K_f, target_shape, device) # Example if K_f could be spatial
        # K_d_grid = prepare_param(K_d, target_shape, device) # Example if K_d could be spatial

        dhdt_physics = calculate_dhdt_physics(
            h=h_pred_grid,
            U=U_grid, # Use potentially grid-based U
            K_f=K_f, # Assuming scalar K_f, m, n, K_d for now
            m=m,
            n=n,
            K_d=K_d,
            dx=dx,
            dy=dy,
            precip=precip,
            da_optimize_params=da_kwargs
        )
    except Exception as e:
        logging.error(f"Error calculating physics term in grid-focused PDE residual: {e}", exc_info=True)
        return h_pred_grid.sum() * 0.0 # Return zero loss

    # 4. 计算残差并返回损失
    pde_residual = dh_dt_grid - dhdt_physics
    return F.mse_loss(pde_residual, torch.zeros_like(pde_residual))


# --- Other Loss Components ---

def compute_conservation_error(model_outputs, physics_params):
    """
    Placeholder for computing mass conservation error.
    Requires careful definition based on boundary conditions and fluxes.
    """
    logging.warning("compute_conservation_error is a placeholder.")
    # Accessing a tensor from model_outputs to get the device correctly
    # Find the first tensor in the dict to determine device
    device = torch.device('cpu')
    ref_tensor = None
    for val in model_outputs.values():
        if isinstance(val, torch.Tensor):
            device = val.device
            ref_tensor = val
            break
    # Return zero loss with grad connection if possible
    if ref_tensor is not None:
        return ref_tensor.sum() * 0.0
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


def compute_smoothness_penalty(predicted_topo, dx=1.0, dy=1.0):
    """
    Computes a penalty based on the gradient magnitude of the predicted topography.
    Assumes predicted_topo is on a grid (B, C, H, W).
    """
    # Use calculate_slope_magnitude which uses Sobel filters via convolution
    # Ensure predicted_topo has the expected 4D shape (B, 1, H, W)
    if predicted_topo.ndim != 4 or predicted_topo.shape[1] != 1:
         logging.warning(f"Smoothness penalty expects input shape (B, 1, H, W), got {predicted_topo.shape}. Skipping penalty.")
         # Return zero tensor with grad
         return predicted_topo.sum() * 0.0

    slope_mag = calculate_slope_magnitude(predicted_topo, dx, dy)
    # Penalize the mean slope magnitude
    smoothness_loss = torch.mean(slope_mag)
    return smoothness_loss


# --- Total Loss Calculation ---

def compute_total_loss(data_pred, final_topo, physics_loss_value, physics_params, loss_weights, collocation_pred=None, collocation_coords=None):
    """Computes the total weighted loss, accepting pre-calculated physics loss.

    Args:
        data_pred (torch.Tensor): Model prediction for data loss (e.g., final state grid).
        final_topo (torch.Tensor): Target data for data loss.
        physics_loss_value (torch.Tensor or None): Pre-calculated physics loss (PDE residual).
        physics_params (dict): Dictionary containing physics parameters (e.g., dx, dy for smoothness).
        loss_weights (dict): Dictionary mapping loss component names to weights.
        collocation_pred (torch.Tensor, optional): Prediction at collocation points (unused here if physics_loss_value is provided).
        collocation_coords (dict, optional): Collocation coordinates (unused here).

    Returns:
        tuple: (total_loss, weighted_losses_dict)
    """
    loss_components = {}
    # Determine device from a valid tensor (prefer data_pred if available)
    if data_pred is not None:
        device = data_pred.device
        ref_tensor_for_grad = data_pred # Use this for zero grad tensor if needed
    elif final_topo is not None:
        device = final_topo.device
        ref_tensor_for_grad = final_topo
    elif physics_loss_value is not None and isinstance(physics_loss_value, torch.Tensor):
         device = physics_loss_value.device
         ref_tensor_for_grad = physics_loss_value
    else:
        device = torch.device('cpu') # Fallback device
        ref_tensor_for_grad = None # Cannot guarantee grad connection

    dx = physics_params.get('dx', 1.0)
    dy = physics_params.get('dy', 1.0)

    # Helper function to create zero tensor with grad connection
    def _zero_with_grad(ref_tensor):
        if ref_tensor is not None and isinstance(ref_tensor, torch.Tensor) and ref_tensor.requires_grad:
            # Multiply by sum to keep grad connection, then zero out
            return (ref_tensor.sum() * 0.0).to(device)
        else:
            # Fallback if no reference tensor is available or it doesn't require grad
            return torch.tensor(0.0, device=device, requires_grad=True)

    # 1. 数据拟合损失
    data_weight = loss_weights.get('data', 0.0)
    if data_weight > 0 and final_topo is not None and data_pred is not None:
        try:
            loss_components['data'] = compute_data_loss(data_pred, final_topo)
        except Exception as e:
             logging.error(f"Error computing data loss: {e}", exc_info=True)
             loss_components['data'] = _zero_with_grad(ref_tensor_for_grad)
    else:
        loss_components['data'] = _zero_with_grad(ref_tensor_for_grad)

    # 2. 物理残差损失 (使用预计算的值)
    physics_weight = loss_weights.get('physics', 0.0)
    if physics_weight > 0 and physics_loss_value is not None and isinstance(physics_loss_value, torch.Tensor) and torch.isfinite(physics_loss_value):
        loss_components['physics'] = physics_loss_value
    else:
        loss_components['physics'] = _zero_with_grad(ref_tensor_for_grad)
        if physics_weight > 0 and (physics_loss_value is None or not isinstance(physics_loss_value, torch.Tensor) or not torch.isfinite(physics_loss_value)):
             logging.warning(f"Invalid or non-finite physics_loss_value received ({physics_loss_value}). Setting physics loss component to zero.")

    # 3. 平滑正则化损失
    smoothness_weight = loss_weights.get('smoothness', 0.0)
    if smoothness_weight > 0 and data_pred is not None:
         try:
            loss_components['smoothness'] = compute_smoothness_penalty(data_pred, dx, dy)
         except Exception as e:
             logging.error(f"Error computing smoothness penalty: {e}", exc_info=True)
             loss_components['smoothness'] = _zero_with_grad(ref_tensor_for_grad)
    else:
        loss_components['smoothness'] = _zero_with_grad(ref_tensor_for_grad)

    # 4. Conservation Loss (Placeholder - ensure grad connection if implemented)
    conservation_weight = loss_weights.get('conservation', 0.0)
    if conservation_weight > 0:
         # Pass data_pred or similar to ensure grad connection if implemented
         loss_components['conservation'] = compute_conservation_error({'data_pred': data_pred}, physics_params)
    else:
         loss_components['conservation'] = _zero_with_grad(ref_tensor_for_grad)


    # 初始化总损失 - 使用首个有效损失作为基础
    weighted_losses = {}
    total_loss = None

    for name, value in loss_components.items():
        weight = loss_weights.get(name, 0.0)
        # Ensure value is a tensor before checking isfinite
        if isinstance(value, torch.Tensor) and weight > 0 and torch.isfinite(value):
            weighted_value = weight * value
            if total_loss is None:
                total_loss = weighted_value
            else:
                # Ensure addition happens between tensors
                total_loss = total_loss + weighted_value

            weighted_losses[f"{name}_loss"] = weighted_value.item() # Log the weighted value
        elif isinstance(value, torch.Tensor) and not torch.isfinite(value):
             logging.warning(f"Non-finite value encountered for loss component '{name}'. Skipping.")
             weighted_losses[f"{name}_loss"] = float('nan')
        else:
             # Log 0 if weight is 0 or value is not a valid tensor for loss
             weighted_losses[f"{name}_loss"] = 0.0

    # 如果没有有效损失项，使用零梯度张量
    if total_loss is None:
        total_loss = _zero_with_grad(ref_tensor_for_grad)

    # Final check for non-finite total_loss
    if not isinstance(total_loss, torch.Tensor) or not torch.isfinite(total_loss):
        logging.error(f"Total loss became non-finite or invalid during accumulation: {total_loss}. Individual weighted losses: {weighted_losses}")
        # 防止NaN传播，使用带梯度的零张量
        total_loss = _zero_with_grad(ref_tensor_for_grad)

    # Ensure total_loss is a tensor before calling .item()
    if isinstance(total_loss, torch.Tensor):
        weighted_losses['total_loss'] = total_loss.item() if torch.isfinite(total_loss) else float('nan')
    else:
        # Handle cases where total_loss might still be None or not a tensor
        weighted_losses['total_loss'] = float('nan')

    return total_loss, weighted_losses


if __name__ == '__main__':
    # Example usage for testing loss functions (needs refinement based on PINN structure)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("Testing loss functions (PDE-based)...")
    batch_size, height, width = 2, 16, 16 # Smaller grid for testing
    N_data = height * width
    N_collocation = 500
    dx, dy = 5.0, 5.0

    # --- Data Points ---
    dummy_data_pred = torch.rand(batch_size, 1, height, width, device='cpu', requires_grad=True) * 100
    dummy_final_topo = torch.rand(batch_size, 1, height, width, device='cpu') * 100

    # --- Collocation Points (for original/adaptive methods) ---
    dummy_collocation_coords = {
        'x': torch.rand(N_collocation, 1, device='cpu', requires_grad=True) * (width - 1) * dx,
        'y': torch.rand(N_collocation, 1, device='cpu', requires_grad=True) * (height - 1) * dy,
        't': torch.rand(N_collocation, 1, device='cpu', requires_grad=True) * 1000.0 # Time requires grad
    }
    dummy_collocation_pred = torch.rand(N_collocation, 1, device='cpu', requires_grad=True) * 100

    # --- Grid Points (for grid-focused method) ---
    dummy_h_pred_grid = torch.rand(batch_size, 1, height, width, device='cpu', requires_grad=True) * 100
    # Example time tensor (scalar, requires grad)
    dummy_t_grid = torch.tensor(500.0, device='cpu', requires_grad=True)

    # --- Parameters ---
    dummy_physics_params = {
        'U': 0.001, 'K_f': 1e-5, 'm': 0.5, 'n': 1.0, 'K_d': 0.01,
        'dx': dx, 'dy': dy, 'epsilon': 1e-12,
        'grid_height': height, 'grid_width': width, # Needed for interpolation method
        'domain_x': [0.0, (width-1)*dx], 'domain_y': [0.0, (height-1)*dy], # Needed for adaptive method normalization
        'drainage_area_kwargs': {'temp': 0.05, 'num_iters': 10} # Params for physics calc
    }
    dummy_loss_weights = {'data': 1.0, 'physics': 0.1, 'conservation': 0.0, 'smoothness': 0.001}

    # --- Test Grid Focused PDE Loss ---
    print("\n--- Testing Grid Focused PDE Loss ---")
    try:
        physics_loss_grid = compute_pde_residual_grid_focused(
            dummy_h_pred_grid, dummy_t_grid, dummy_physics_params
        )
        print(f"Grid Focused Physics Loss: {physics_loss_grid.item():.6f}")

        # Test total loss using grid-focused physics loss
        total_loss_grid, loss_dict_grid = compute_total_loss(
            data_pred=dummy_data_pred, # Use separate data prediction
            final_topo=dummy_final_topo,
            physics_loss_value=physics_loss_grid, # Pass the pre-calculated grid loss
            physics_params=dummy_physics_params,
            loss_weights=dummy_loss_weights
        )
        print(f"Total Loss (using grid physics): {total_loss_grid.item():.6f}")
        print(f"Loss Components (grid physics): {loss_dict_grid}")

        # Test backward pass
        if isinstance(total_loss_grid, torch.Tensor) and torch.isfinite(total_loss_grid):
            print("Testing backward pass (grid physics)...")
            total_loss_grid.backward()
            print("Backward pass successful.")
            # Check grad on the grid prediction
            print("Gradient on dummy_h_pred_grid exists:", dummy_h_pred_grid.grad is not None)
        else:
            print("Skipping backward pass due to non-finite/invalid total loss.")

    except Exception as e:
        print(f"Error during grid focused test: {e}")
        import traceback
        traceback.print_exc()


    # --- Test Original Interpolation PDE Loss ---
    print("\n--- Testing Interpolation PDE Loss ---")
    try:
        # Reset grads if needed
        if dummy_collocation_pred.grad is not None: dummy_collocation_pred.grad.zero_()
        if dummy_data_pred.grad is not None: dummy_data_pred.grad.zero_()

        physics_loss_interp = compute_pde_residual(
            dummy_collocation_pred, dummy_collocation_coords, dummy_physics_params
        )
        print(f"Interpolation Physics Loss: {physics_loss_interp.item():.6f}")

        # Test total loss using interpolation-based physics loss
        total_loss_interp, loss_dict_interp = compute_total_loss(
            data_pred=dummy_data_pred,
            final_topo=dummy_final_topo,
            physics_loss_value=physics_loss_interp, # Pass the pre-calculated interp loss
            physics_params=dummy_physics_params,
            loss_weights=dummy_loss_weights,
            collocation_pred=dummy_collocation_pred # Pass these for reference if needed by other losses
        )
        print(f"Total Loss (using interp physics): {total_loss_interp.item():.6f}")
        print(f"Loss Components (interp physics): {loss_dict_interp}")

        # Test backward pass
        if isinstance(total_loss_interp, torch.Tensor) and torch.isfinite(total_loss_interp):
            print("Testing backward pass (interp physics)...")
            # Need to retain graph if backward is called multiple times on same graph parts
            # total_loss_interp.backward(retain_graph=True) # Might be needed if graphs overlap significantly
            total_loss_interp.backward()
            print("Backward pass successful.")
            # Check grad on the collocation prediction
            print("Gradient on dummy_collocation_pred exists:", dummy_collocation_pred.grad is not None)
        else:
            print("Skipping backward pass due to non-finite/invalid total loss.")

    except Exception as e:
        print(f"Error during interpolation test: {e}")
        import traceback
        traceback.print_exc()



# --- PDE Residual Calculation (Dual Output Model) ---

def compute_pde_residual_dual_output(outputs, physics_params):
    """使用模型直接输出的状态和导数计算PDE残差

    Args:
        outputs (dict): 模型输出字典，必须包含 'state' 和 'derivative' 张量。
                        'state' shape: [B, C_out, H, W] or [N, C_out]
                        'derivative' shape: [B, C_out, H, W] or [N, C_out]
        physics_params (dict): 物理参数字典。

    Returns:
        torch.Tensor: PDE残差均方误差。
    """
    if 'state' not in outputs or 'derivative' not in outputs:
        raise ValueError("模型输出字典必须包含 'state' 和 'derivative' 键。")

    h_pred = outputs['state']
    dh_dt_pred = outputs['derivative']

    # 检查输入形状是否一致
    if h_pred.shape != dh_dt_pred.shape:
        raise ValueError(f"状态和导数预测的形状不匹配: state={h_pred.shape}, derivative={dh_dt_pred.shape}")

    # 根据输入形状判断是网格模式还是坐标点模式
    is_grid_mode = h_pred.ndim == 4

    try:
        # 提取必要的物理参数
        U = physics_params.get('U', 0.0)
        K_f = physics_params.get('K_f', 1e-5)
        m = physics_params.get('m', 0.5)
        n = physics_params.get('n', 1.0)
        K_d = physics_params.get('K_d', 0.01)
        dx = physics_params.get('dx', 1.0)
        dy = physics_params.get('dy', 1.0)
        precip = physics_params.get('precip', 1.0)
        da_kwargs = physics_params.get('drainage_area_kwargs', {})
        epsilon = physics_params.get('epsilon', 1e-10)

        # --- 计算物理倾向项 dh/dt_physics --- 
        if is_grid_mode:
            # 网格模式: 直接使用 calculate_dhdt_physics
            # 注意：这里假设 calculate_dhdt_physics 内部使用了正确的导数计算
            # (理想情况下是使用我们之前验证过的自定义导数函数)
            from .physics import calculate_dhdt_physics # 确保导入

            # 处理参数的空间变化 (如果需要)
            device = h_pred.device
            target_shape = h_pred.shape
            # (复用 prepare_param 逻辑，或者直接在 calculate_dhdt_physics 中处理)
            def prepare_param(param_val, target_shape, device):
                 if isinstance(param_val, torch.Tensor):
                     param_val = param_val.to(device)
                     if param_val.shape == target_shape: return param_val
                     elif param_val.numel() == 1: return param_val.expand(target_shape)
                     elif param_val.ndim == 1 and param_val.shape[0] == target_shape[0]: return param_val.view(-1, 1, 1, 1).expand(target_shape)
                     else:
                         try: return param_val.expand(target_shape)
                         except RuntimeError: raise ValueError(f"无法广播参数形状 {param_val.shape} 到目标 {target_shape}")
                 else: return torch.full(target_shape, float(param_val), device=device)
            
            U_grid = prepare_param(U, target_shape, device)
            # K_f_grid = prepare_param(K_f, target_shape, device) # 如果 K_f 是空间的
            # K_d_grid = prepare_param(K_d, target_shape, device) # 如果 K_d 是空间的

            dhdt_physics = calculate_dhdt_physics(
                h=h_pred,
                U=U_grid, # 使用可能的网格参数
                K_f=K_f, # 假设标量
                m=m,
                n=n,
                K_d=K_d, # 假设标量
                dx=dx,
                dy=dy,
                precip=precip,
                da_optimize_params=da_kwargs
            )
        else:
            # 坐标点模式: 当前实现依赖 compute_local_physics，其物理计算与 physics.py 不一致。
            # 考虑到主要应用场景是网格数据，暂时禁用或警告此模式。
            logging.error("compute_pde_residual_dual_output currently primarily supports grid mode input (4D tensors). "
                          "The coordinate point mode relies on compute_local_physics which has known inconsistencies "
                          "with the main physics module. This path needs review or refactoring.")
            # Option 1: Raise an error to prevent usage
            # raise NotImplementedError("compute_pde_residual_dual_output for coordinate points needs refactoring "
            #                           "to ensure consistent physics calculation.")
            # Option 2: Return zero loss with warning (less strict)
            logging.warning("Returning zero physics loss for dual_output in coordinate mode due to implementation inconsistencies.")
            # Directly return zero loss for coordinate mode as intended by the warning
            return torch.tensor(0.0, device=h_pred.device, dtype=h_pred.dtype, requires_grad=False) # Ensure no grad

        # --- 计算残差 (Only for Grid Mode now) ---
        pde_residual = dh_dt_pred - dhdt_physics

        # 返回均方误差损失 (Only for Grid Mode now)
        return F.mse_loss(pde_residual, torch.zeros_like(pde_residual))

    except Exception as e:
        logging.error(f"计算双输出 PDE 残差时出错: {e}", exc_info=True)
        # 返回一个零损失，但保留梯度连接（如果可能）
        zero_loss = (outputs['state'].sum() + outputs['derivative'].sum()) * 0.0
        return zero_loss


    print("\nLoss functions testing done.")
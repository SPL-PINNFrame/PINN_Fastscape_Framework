import torch
import torch.optim as optim
import logging
from tqdm import tqdm
import time
import numpy as np
import torch.nn.functional as F # Ensure F is imported if used elsewhere
import logging
import torch
from torch import optim
# ADDED: Import calculate_laplacian
from src.physics import calculate_laplacian
from scipy.interpolate import griddata
import torch.nn.functional as F
import os # For save_path
# from scipy.optimize import minimize as scipy_minimize # Comment out SciPy import

# --- Interpolation Function ---

def interpolate_uplift_cv(uplift_params_flat, param_shape, target_shape, method='linear'):
    """
    Interpolates low-resolution uplift parameters to the target grid shape.
    Uses scipy.interpolate.griddata (suitable for SciPy optimizers).

    Args:
        uplift_params_flat (np.ndarray): Flattened array of uplift parameters.
        param_shape (tuple): Shape of the low-resolution parameter grid (e.g., (10, 10)).
        target_shape (tuple): Shape of the target high-resolution grid (e.g., (100, 100)).
        method (str): Interpolation method ('linear', 'nearest', 'cubic').

    Returns:
        np.ndarray: Interpolated uplift field with target_shape.
    """
    param_grid = uplift_params_flat.reshape(param_shape)
    param_h, param_w = param_shape
    target_h, target_w = target_shape

    # Create coordinates for the low-resolution parameter grid
    param_x = np.linspace(0, 1, param_w)
    param_y = np.linspace(0, 1, param_h)
    param_points = np.array([[x, y] for y in param_y for x in param_x])

    # Create coordinates for the high-resolution target grid
    target_x = np.linspace(0, 1, target_w)
    target_y = np.linspace(0, 1, target_h)
    target_grid_y, target_grid_x = np.meshgrid(target_y, target_x, indexing='ij')
    target_points = np.stack([target_grid_x.ravel(), target_grid_y.ravel()], axis=-1)

    # Interpolate
    try:
        interpolated_uplift_flat = griddata(param_points, uplift_params_flat, target_points, method=method, fill_value=np.mean(uplift_params_flat))
        # Handle potential NaNs if method='cubic'
        interpolated_uplift_flat = np.nan_to_num(interpolated_uplift_flat, nan=np.mean(uplift_params_flat))
    except Exception as e:
        logging.error(f"Interpolation failed: {e}. Returning mean value grid.")
        interpolated_uplift_flat = np.full(target_points.shape[0], np.mean(uplift_params_flat))


    return interpolated_uplift_flat.reshape(target_shape)



def interpolate_uplift_torch(uplift_params, param_shape, target_shape, method='rbf', sigma=0.1):
    """可微分的插值函数，用于PyTorch优化路径
    
    Args:
        uplift_params (torch.Tensor): 需要插值的参数张量 (flattened or grid shape)
        param_shape (tuple): 参数张量的原始形状 (H_param, W_param)
        target_shape (tuple): 目标形状 (H_target, W_target)
        method (str): 插值方法 ('rbf', 'bilinear')
        sigma (float): RBF插值的带宽参数
        
    Returns:
        torch.Tensor: 插值后的张量, shape=target_shape
    """
    device = uplift_params.device
    
    # Ensure input is flattened
    if uplift_params.ndim != 1:
        uplift_params_flat = uplift_params.flatten()
    else:
        uplift_params_flat = uplift_params
        
    param_h, param_w = param_shape
    target_h, target_w = target_shape
    
    # 创建归一化的源网格坐标
    x_src = torch.linspace(0, 1, param_w, device=device)
    y_src = torch.linspace(0, 1, param_h, device=device)
    grid_y_src, grid_x_src = torch.meshgrid(y_src, x_src, indexing='ij')
    points_src = torch.stack([grid_x_src.flatten(), grid_y_src.flatten()], dim=1) # Shape (H_param*W_param, 2)
    
    # 创建归一化的目标网格坐标
    x_tgt = torch.linspace(0, 1, target_w, device=device)
    y_tgt = torch.linspace(0, 1, target_h, device=device)
    grid_y_tgt, grid_x_tgt = torch.meshgrid(y_tgt, x_tgt, indexing='ij')
    points_tgt = torch.stack([grid_x_tgt.flatten(), grid_y_tgt.flatten()], dim=1) # Shape (H_target*W_target, 2)
    
    # 获取源值 (ensure it's flat)
    values_src = uplift_params_flat # Shape (H_param*W_param,)
    
    if method == 'rbf':
        # 使用RBF插值
        # 计算点对点距离矩阵
        diff = points_tgt.unsqueeze(1) - points_src.unsqueeze(0) # [Q, P, 2]
        dist_sq = torch.sum(diff**2, dim=2)  # [Q, P]
        
        # 计算RBF权重
        weights = torch.exp(-dist_sq / (2 * sigma**2))  # [Q, P]
        weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-10)
        
        # 加权平均计算插值结果
        # Ensure values_src is [P, 1] for matmul
        values_tgt = torch.matmul(weights, values_src.unsqueeze(1)).squeeze(1) # [Q, P] @ [P, 1] -> [Q, 1] -> [Q]
    elif method == 'bilinear':
        # 使用grid_sample实现双线性插值
        # 需要将目标坐标从[0,1]映射到[-1,1] for grid_sample
        grid_x_norm_gs = 2.0 * grid_x_tgt - 1.0
        grid_y_norm_gs = 2.0 * grid_y_tgt - 1.0
        # grid_sample expects grid shape (N, H_out, W_out, 2)
        grid_sample_coords = torch.stack([grid_x_norm_gs, grid_y_norm_gs], dim=2).unsqueeze(0) # Shape (1, H_target, W_target, 2)
        
        # 重塑参数网格并添加批次和通道维度
        param_grid = uplift_params_flat.reshape(1, 1, param_h, param_w) # Shape (1, 1, H_param, W_param)
        
        # 使用grid_sample进行插值
        # align_corners=True is often recommended for resolution changes
        # Ensure inputs to grid_sample are float32
        values_tgt_grid = torch.nn.functional.grid_sample(
            param_grid.float(),  # Convert input grid to float
            grid_sample_coords.float(), # Convert sampling coordinates to float
            mode='bilinear',
            padding_mode='border', # or 'zeros', 'reflection'
            align_corners=True
        )
        values_tgt = values_tgt_grid.squeeze() # Remove B, C dims -> Shape (H_target, W_target)
        # Flatten to match RBF output shape if needed, but returning grid is more useful
        # values_tgt = values_tgt.flatten() 
        return values_tgt # Return grid shape (H_target, W_target)
    else:
        raise ValueError(f"未知的插值方法: {method}")
    
    # Reshape RBF result to target grid shape
    return values_tgt.reshape(target_shape)

# --- NEW: Parameter Optimizer Class (Grid Focused) ---

class ParameterOptimizer:
    """参数优化器，专为地形网格优化设计"""

    def __init__(self, model, observation_data, initial_state=None, fixed_params=None, t_target=1.0):
        """
        Args:
            model (torch.nn.Module): 训练好的PINN模型 (应支持 predict_state 模式).
            observation_data (torch.Tensor): 观测到的地形数据 [B, 1, H, W].
            initial_state (torch.Tensor, optional): 初始地形 [B, 1, H, W]. 如果为 None, 使用零初始化.
            fixed_params (dict, optional): 固定的物理参数字典 (例如 {'K': val, 'D': val}).
            t_target (float or torch.Tensor): 观测数据对应的时间点.
        """
        self.model = model
        self.observation = observation_data
        self.device = observation_data.device
        self.dtype = observation_data.dtype

        # Ensure model is on the correct device and in eval mode
        self.model.to(self.device)
        self.model.eval()

        # Handle initial state
        if initial_state is None:
            logging.info("Initial state not provided, using zeros.")
            self.initial_state = torch.zeros_like(observation_data, device=self.device, dtype=self.dtype)
        else:
            self.initial_state = initial_state.to(device=self.device, dtype=self.dtype)

        # Store fixed parameters and target time
        self.fixed_params = fixed_params if fixed_params is not None else {}
        # Ensure fixed params are tensors on the correct device
        for k, v in self.fixed_params.items():
             if not isinstance(v, torch.Tensor):
                  # Attempt to convert scalar to tensor matching observation shape (B, 1, H, W)
                  try:
                       self.fixed_params[k] = torch.full_like(observation_data, float(v), device=self.device, dtype=self.dtype)
                       logging.debug(f"Converted scalar fixed parameter '{k}' to tensor.")
                  except ValueError:
                       logging.error(f"Cannot convert fixed parameter '{k}' to tensor. Please provide tensors.")
                       raise
             else:
                  self.fixed_params[k] = v.to(device=self.device, dtype=self.dtype)


        # Handle target time
        if isinstance(t_target, (int, float)):
             self.t_target = torch.tensor(float(t_target), device=self.device, dtype=self.dtype)
        elif isinstance(t_target, torch.Tensor):
             self.t_target = t_target.to(device=self.device, dtype=self.dtype)
        else:
             raise TypeError(f"Unsupported type for t_target: {type(t_target)}")


        # Extract grid dimensions
        self.batch_size, _, self.height, self.width = observation_data.shape
        logging.info(f"ParameterOptimizer initialized for grid shape: B={self.batch_size}, H={self.height}, W={self.width}")

    def _ensure_initial_param_shape(self, initial_value, param_name):
        """确保初始参数张量具有正确的形状 [B, 1, H, W]。"""
        target_shape = (self.batch_size, 1, self.height, self.width)
        if initial_value is None:
            # Default: Initialize with ones (can be scaled later if needed)
            logging.info(f"No initial value provided for '{param_name}'. Initializing with ones.")
            param_tensor = torch.ones(target_shape, device=self.device, dtype=self.dtype, requires_grad=True)
        elif isinstance(initial_value, (int, float)):
            # Scalar initial value
            param_tensor = torch.full(target_shape, float(initial_value), device=self.device, dtype=self.dtype, requires_grad=True)
        elif isinstance(initial_value, torch.Tensor):
            param_tensor = initial_value.clone().to(device=self.device, dtype=self.dtype)
            # Adjust shape if necessary
            if param_tensor.shape != target_shape:
                if param_tensor.numel() == 1: # Scalar tensor
                    param_tensor = param_tensor.expand(target_shape).clone().detach().requires_grad_(True)
                elif param_tensor.ndim == 2 and param_tensor.shape == (self.height, self.width): # H, W
                    param_tensor = param_tensor.unsqueeze(0).unsqueeze(0).expand(target_shape).clone().detach().requires_grad_(True)
                elif param_tensor.ndim == 3 and param_tensor.shape == (self.batch_size, self.height, self.width): # B, H, W
                    param_tensor = param_tensor.unsqueeze(1).clone().detach().requires_grad_(True)
                else:
                    # Attempt interpolation if shapes don't match exactly but dims are compatible
                    logging.warning(f"Initial value shape {param_tensor.shape} for '{param_name}' doesn't match target {target_shape}. Attempting interpolation.")
                    try:
                        # Ensure 4D input for interpolate
                        if param_tensor.ndim == 2: param_tensor = param_tensor.unsqueeze(0).unsqueeze(0)
                        elif param_tensor.ndim == 3: param_tensor = param_tensor.unsqueeze(1)

                        param_tensor = F.interpolate(param_tensor, size=(self.height, self.width), mode='bilinear', align_corners=False)
                        # Ensure correct batch size
                        if param_tensor.shape[0] != self.batch_size:
                             param_tensor = param_tensor.expand(self.batch_size, -1, -1, -1)
                        param_tensor = param_tensor.clone().detach().requires_grad_(True)
                        logging.info(f"Successfully interpolated initial value for '{param_name}' to {param_tensor.shape}.")
                    except Exception as e:
                        logging.error(f"Failed to interpolate initial value for '{param_name}': {e}")
                        raise ValueError(f"Initial value shape {initial_value.shape} for '{param_name}' cannot be adapted to target {target_shape}.")
            else:
                 # Ensure requires_grad is True even if shape matches
                 if not param_tensor.requires_grad:
                      param_tensor.requires_grad_(True)
        else:
            raise TypeError(f"Unsupported type for initial_value of '{param_name}': {type(initial_value)}")

        return param_tensor


    def create_objective_function(self, params_to_optimize, spatial_smoothness=0.0, bounds=None):
        """创建优化目标函数 (用于 PyTorch 优化器).

        Args:
            params_to_optimize (dict): 包含待优化参数张量的字典 {'param_name': tensor}. Tensor requires grad.
            spatial_smoothness (float): 空间平滑度正则化权重.
            bounds (dict, optional): 参数边界字典 {'param_name': (min, max)}.

        Returns:
            callable: 优化目标函数，返回 (total_loss, loss_components_dict).
        """
        param_names = list(params_to_optimize.keys())
        logging.info(f"Creating objective function to optimize: {param_names}")
        if spatial_smoothness > 0:
             logging.info(f"Applying spatial smoothness regularization with weight: {spatial_smoothness}")
        if bounds:
             logging.info(f"Applying bounds: {bounds}")

        # Create parameter constraint function
        def constrain_parameters(param_dict):
            constrained_dict = {}
            if bounds:
                for name, param in param_dict.items():
                    if name in bounds:
                        min_val, max_val = bounds[name]
                        constrained_dict[name] = torch.clamp(param, min=min_val, max=max_val)
                    else:
                        constrained_dict[name] = param
                return constrained_dict
            else:
                return param_dict # No constraints

        def objective_function():
            # Apply constraints (clamping) - this is done outside backward pass usually
            # For optimization step, we use the raw tensor, apply constraints after step if needed
            # However, for the forward pass *during* optimization, use constrained values if bounds exist
            current_params_constrained = constrain_parameters(params_to_optimize)

            # Combine fixed and currently optimized parameters
            all_params = {**self.fixed_params, **current_params_constrained}

            # Prepare model input
            model_input = {
                'initial_state': self.initial_state,
                'params': all_params,
                't_target': self.t_target
            }

            # Use model to predict final state
            # Ensure model is in eval mode if not done globally
            # self.model.eval()
            predicted_state = self.model(model_input, mode='predict_state')

            # Calculate data fidelity loss
            data_loss = F.mse_loss(predicted_state, self.observation)
            loss_components = {'data_loss': data_loss.item()}
            total_loss = data_loss

            # Add spatial smoothness regularization (Laplacian penalty)
            if spatial_smoothness > 0:
                smoothness_loss_total = torch.tensor(0.0, device=self.device, dtype=self.dtype) # Initialize as tensor
                for name, param in params_to_optimize.items():
                    # MODIFIED: Use calculate_laplacian function
                    # Assumes dx=1, dy=1 for simplicity in penalty term,
                    # or get dx, dy from self.physics_params if available and needed.
                    # Using dx=1, dy=1 means we penalize the curvature directly.
                    laplacian = calculate_laplacian(param, dx=1.0, dy=1.0)

                    # Penalize the squared magnitude of the Laplacian
                    smoothness_loss = torch.mean(laplacian**2) * spatial_smoothness
                    smoothness_loss_total = smoothness_loss_total + smoothness_loss # Add tensors
                    loss_components[f'{name}_smoothness_loss'] = smoothness_loss.item()
                total_loss = total_loss + smoothness_loss_total
                # Ensure smoothness_loss_total is stored as item if it's still a tensor
                loss_components['smoothness_loss'] = smoothness_loss_total.item()


            loss_components['total_loss'] = total_loss.item()
            return total_loss, loss_components

        return objective_function


# --- NEW: PyTorch-based Optimization Function ---

def optimize_parameters(model, observation_data, params_to_optimize_config, config,
                       initial_state=None, fixed_params=None, t_target=1.0):
    """运行参数优化流程 (使用 PyTorch 优化器).

    Args:
        model (torch.nn.Module): 训练好的 PINN 模型.
        observation_data (torch.Tensor): 观测地形数据 [B, 1, H, W].
        params_to_optimize_config (dict): 配置要优化的参数
                                          {'param_name': {'initial_value': val, 'bounds': (min, max), ...}, ...}.
        config (dict): 主配置文件，包含 'optimization_params'.
        initial_state (torch.Tensor, optional): 初始地形状态 [B, 1, H, W].
        fixed_params (dict, optional): 固定的物理参数字典.
        t_target (float or torch.Tensor): 观测数据对应的时间点.

    Returns:
        tuple: (optimized_params_dict, history)
               - optimized_params_dict: 包含优化后参数张量的字典.
               - history: 包含优化历史（如损失）的字典.
    """
    opt_config = config.get('optimization_params', {})
    optimizer_name = opt_config.get('optimizer', 'Adam').lower()
    lr = opt_config.get('learning_rate', 0.01)
    iterations = opt_config.get('max_iterations', 1000)
    spatial_smoothness = opt_config.get('spatial_smoothness_weight', 0.0)

    # --- 初始化 ParameterOptimizer ---
    param_optimizer_instance = ParameterOptimizer(model, observation_data, initial_state, fixed_params, t_target)
    device = param_optimizer_instance.device

    # --- 初始化待优化参数 ---
    params_to_optimize = {}
    param_bounds = {}
    for name, p_config in params_to_optimize_config.items():
        initial_value = p_config.get('initial_value', None)
        bounds = p_config.get('bounds', None)
        params_to_optimize[name] = param_optimizer_instance._ensure_initial_param_shape(initial_value, name)
        if bounds:
            param_bounds[name] = bounds

    # --- 创建优化目标函数 ---
    objective_fn = param_optimizer_instance.create_objective_function(
        params_to_optimize, spatial_smoothness, param_bounds
    )

    # --- 创建 PyTorch 优化器 ---
    params_list = list(params_to_optimize.values())
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(params_list, lr=lr, betas=opt_config.get('betas', (0.9, 0.999)))
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(params_list, lr=lr, weight_decay=opt_config.get('weight_decay', 0.01))
    elif optimizer_name == 'lbfgs':
        # LBFGS 需要 closure
        optimizer = torch.optim.LBFGS(params_list, lr=lr, max_iter=opt_config.get('lbfgs_max_iter', 20), line_search_fn="strong_wolfe")
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_name}")

    logging.info(f"开始使用 {optimizer_name} 进行参数优化，共 {iterations} 次迭代...")
    history = {'loss': [], 'iterations': 0, 'time': 0.0, 'loss_components': []}
    start_time = time.time()

    # --- 优化循环 ---
    for i in tqdm(range(iterations), desc="优化参数"):
        def closure():
            optimizer.zero_grad()
            loss, loss_components = objective_fn()
            if torch.is_tensor(loss) and loss.requires_grad:
                loss.backward()
            elif not torch.is_tensor(loss):
                 logging.error(f"目标函数未返回张量损失。")
                 # LBFGS 需要返回损失值
                 return torch.tensor(float('nan'), device=device)
            # 记录损失组件（在 closure 内部，可能被 LBFGS 多次调用）
            # history['loss_components'].append(loss_components) # 可能导致记录过多
            return loss

        # --- Optimizer Step ---
        if optimizer_name == 'lbfgs':
            loss = optimizer.step(closure) # LBFGS step
        else:
            optimizer.zero_grad()
            loss, loss_components = objective_fn() # Calculate loss and grads
            if torch.is_tensor(loss) and torch.isfinite(loss) and loss.requires_grad:
                loss.backward()
                optimizer.step() # Adam/AdamW step
            elif not torch.is_tensor(loss) or not torch.isfinite(loss):
                 logging.warning(f"迭代 {i}: 损失无效 ({loss}). 跳过优化步骤。")
                 # Optionally break if loss becomes NaN/Inf
                 if not torch.isfinite(loss):
                      logging.error("优化因损失无效而停止。")
                      break

        # --- 应用边界约束 (在优化步骤之后) ---
        if param_bounds:
            with torch.no_grad():
                for name, param in params_to_optimize.items():
                    if name in param_bounds:
                        min_val, max_val = param_bounds[name]
                        param.clamp_(min=min_val, max=max_val)

        # --- 记录历史 ---
        current_loss = loss.item() if torch.is_tensor(loss) else float('nan')
        history['loss'].append(current_loss)
        # Log components from the last objective call (may not be exact for LBFGS multi-eval)
        if 'loss_components' not in locals(): # If Adam/W didn't run objective this iter
             _, loss_components = objective_fn() # Recalculate for logging
        history['loss_components'].append(loss_components)


        if i % opt_config.get('log_interval', 50) == 0:
            logging.info(f"迭代 {i}/{iterations}, 损失: {current_loss:.6e}")
            # Log component details less frequently
            if i % (opt_config.get('log_interval', 50) * 5) == 0:
                 log_str = ", ".join([f"{k}: {v:.3e}" for k, v in loss_components.items() if k != 'total_loss'])
                 logging.info(f"  Components: {log_str}")


        # --- 收敛检查 (Adam/AdamW) ---
        if optimizer_name != 'lbfgs':
            if i > opt_config.get('convergence_patience', 20): # Check after some iterations
                 loss_hist = history['loss'][-opt_config.get('convergence_patience', 20):]
                 if len(loss_hist) > 1 and np.abs(loss_hist[-1] - np.mean(loss_hist[:-1])) < opt_config.get('loss_tolerance', 1e-7):
                      logging.info(f"在迭代 {i} 时因损失变化小而收敛。")
                      break
            if not np.isfinite(current_loss):
                 logging.error(f"在迭代 {i} 时损失无效。停止优化。")
                 break

    # --- 结束 ---
    end_time = time.time()
    total_time = end_time - start_time
    history['iterations'] = len(history['loss'])
    history['time'] = total_time
    history['final_loss'] = history['loss'][-1] if history['loss'] else float('nan')

    logging.info(f"优化在 {total_time:.2f} 秒内完成 {history['iterations']} 次迭代。")
    logging.info(f"最终损失: {history['final_loss']:.6e}")

    # Detach final parameters
    optimized_detached = {k: v.detach().clone() for k, v in params_to_optimize.items()}

    # --- 保存结果 ---
    save_path = opt_config.get('save_path', None)
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_dict = {
                'optimized_params': {k: v.cpu() for k,v in optimized_detached.items()}, # Save to CPU
                'history': history,
                'config': config # Save the full config used
            }
            torch.save(save_dict, save_path)
            logging.info(f"优化结果已保存到: {save_path}")
        except Exception as e:
            logging.error(f"保存优化结果失败: {e}")

    return optimized_detached, history


# --- Similarity Function ---

def terrain_similarity(pred_topo_np, target_topo_np, metric='mse'):
    """
    Calculates similarity (loss) between predicted and target topography.

    Args:
        pred_topo_np (np.ndarray): Predicted topography grid.
        target_topo_np (np.ndarray): Target topography grid.
        metric (str): Similarity metric ('mse', 'mae').

    Returns:
        float: Calculated similarity loss.
    """
    if pred_topo_np.shape != target_topo_np.shape:
        # Attempt to resize prediction if shapes mismatch (e.g., due to padding)
        # This might indicate an upstream issue.
        logging.warning(f"Shape mismatch in terrain_similarity: pred={pred_topo_np.shape}, target={target_topo_np.shape}. Attempting resize.")
        # Example using simple cropping/padding - adjust as needed
        h_diff = pred_topo_np.shape[0] - target_topo_np.shape[0]
        w_diff = pred_topo_np.shape[1] - target_topo_np.shape[1]
        if h_diff > 0: pred_topo_np = pred_topo_np[h_diff//2:-(h_diff - h_diff//2), :]
        if w_diff > 0: pred_topo_np = pred_topo_np[:, w_diff//2:-(w_diff - w_diff//2)]
        # Add padding if needed (less common)

        # If still mismatch after simple adjustment, raise error
        if pred_topo_np.shape != target_topo_np.shape:
             raise ValueError(f"Cannot reconcile shapes for similarity calculation: pred={pred_topo_np.shape}, target={target_topo_np.shape}")


    if metric == 'mse':
        return np.mean((pred_topo_np - target_topo_np)**2)
    elif metric == 'mae':
        return np.mean(np.abs(pred_topo_np - target_topo_np))
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")

# --- Objective Function Creation (Old - SciPy based) ---
# Note: Commenting out the SciPy-based objective creation and optimization functions
# as the new PyTorch-based approach (`ParameterOptimizer`, `optimize_parameters`) is preferred.
# Keep interpolation and similarity functions for potential reuse or reference.

# def create_objective_function_pinn(
#     target_dem_np,
#     pinn_model,
#     model_inputs_template, # Dict containing fixed inputs like initial_topo, K, D etc.
#     uplift_param_shape,
#     target_dem_shape, # Should match target_dem_np.shape
#     device,
#     interpolation_fn=interpolate_uplift_cv, # Function to interpolate uplift
#     similarity_fn=terrain_similarity, # Function to compare terrains
#     opt_config=None # Pass optimization config for regularization etc.
# ):
#     """
#     (Commented out - Replaced by PyTorch-based ParameterOptimizer)
#     Creates the objective function specifically for uplift parameter optimization using PINN.
#     ... (original docstring) ...
#     """
#     pinn_model.eval() # Ensure model is in evaluation mode
#     target_dem_torch = torch.from_numpy(target_dem_np).float().unsqueeze(0).unsqueeze(0).to(device) # Add B, C dims
#
#     # Extract fixed parameters (K, D) from template
#     fixed_params = {
#         'K': model_inputs_template.get('k_sp'), # Assuming k_sp corresponds to K
#         'D': model_inputs_template.get('k_d')  # Assuming k_d corresponds to D
#     }
#     if fixed_params['K'] is None or fixed_params['D'] is None:
#          logging.warning("Missing fixed parameters K (k_sp) or D (k_d) in model_inputs_template.")
#          pass # Allow proceeding if model handles None, otherwise raise error earlier
#
#     initial_topo = model_inputs_template.get('initial_topography')
#     if initial_topo is None:
#         raise ValueError("Missing 'initial_topography' in model_inputs_template.")
#
#     t_target = model_inputs_template.get('t_target') # Get target time directly from template
#     if t_target is None:
#          raise ValueError("Missing target time 't_target' or 'physics_params.total_time'.")
#
#     # --- Regularization ---
#     reg_weight = 0.0
#     if opt_config:
#          reg_weight = opt_config.get('parameter_regularization_weight', 0.0)
#
#     def objective_function(uplift_params_flat_np):
#         """The actual objective function called by the optimizer."""
#         # 1. Interpolate
#         uplift_grid_np = interpolation_fn(uplift_params_flat_np, uplift_param_shape, target_dem_shape)
#         uplift_grid_torch = torch.from_numpy(uplift_grid_np).float().unsqueeze(0).unsqueeze(0).to(device) # Add B, C dims
#         # 2. Prepare inputs
#         current_params = {**fixed_params, 'U': uplift_grid_torch}
#         model_input_tuple = (initial_topo, current_params, t_target)
#         # 3. Run PINN prediction
#         try:
#             with torch.no_grad():
#                 pred_topo_torch = pinn_model(x=model_input_tuple, mode='predict_state')
#             pred_topo_np = pred_topo_torch.squeeze().cpu().numpy()
#         except Exception as e:
#              logging.error(f"Error during PINN model prediction in objective function: {e}")
#              return 1e10 # Return a large value on error
#         # 4. Calculate similarity loss
#         similarity_loss = similarity_fn(pred_topo_np, target_dem_np)
#         # 5. Add regularization loss (optional)
#         regularization_loss = 0.0
#         if reg_weight > 0:
#              regularization_loss = reg_weight * np.mean(uplift_params_flat_np**2)
#         total_loss = similarity_loss + regularization_loss
#         return total_loss
#
#     return objective_function


# --- SciPy Optimization Wrapper (Old - Commented Out) ---

# def optimize_uplift_scipy(
#     objective_func,
#     initial_uplift_params_flat,
#     bounds=None,
#     method='L-BFGS-B',
#     max_iter=100,
#     **scipy_options # Pass other options like tol, disp, etc.
# ):
#     """
#     (Commented out - Replaced by PyTorch-based optimize_parameters)
#     Wrapper for running uplift optimization using SciPy optimizers.
#     ... (original docstring) ...
#     """
#     logging.info(f"Starting SciPy optimization using method '{method}' for max {max_iter} iterations.")
#     options = {'maxiter': max_iter, 'disp': scipy_options.get('disp', True)}
#     options.update({k: v for k, v in scipy_options.items() if k not in ['disp']}) # Add other options
#     start_time = time.time()
#     result = scipy_minimize(
#         fun=objective_func,
#         x0=initial_uplift_params_flat,
#         method=method,
#         bounds=bounds,
#         options=options
#     )
#     end_time = time.time()
#     logging.info(f"SciPy optimization finished in {end_time - start_time:.2f} seconds.")
#     logging.info(f"Success: {result.success}, Status: {result.message}, Final Loss: {result.fun:.6f}, Iterations: {result.nit}")
#     return result


# --- PyTorch Optimization Wrapper (Placeholder/Example) ---

def optimize_uplift_torch(
    pinn_model,
    target_dem, # Torch tensor on device
    model_inputs_template, # Dict with fixed inputs on device
    initial_uplift_params, # Torch tensor, requires_grad=True
    target_dem_shape,
    device,
    interpolation_fn, # Needs to be a *differentiable* torch function
    similarity_fn, # Needs to be a *differentiable* torch function
    optimizer_name='Adam',
    lr=0.01,
    max_iter=100,
    bounds=None, # Tuple (min, max) for parameter values
    opt_config=None
):
    """
    Placeholder for running uplift optimization using PyTorch optimizers.
    Requires differentiable interpolation and similarity functions.
    """
    logging.warning("optimize_uplift_torch requires differentiable interpolation_fn and similarity_fn.")
    logging.info(f"Starting PyTorch optimization using {optimizer_name} for {max_iter} iterations.")

    if not initial_uplift_params.requires_grad:
        initial_uplift_params.requires_grad_(True)

    params_to_optimize = [initial_uplift_params]

    # Setup optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(params_to_optimize, lr=lr)
    elif optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(params_to_optimize, lr=lr)
    elif optimizer_name.lower() == 'lbfgs':
        optimizer = optim.LBFGS(params_to_optimize, lr=lr, max_iter=20) # LBFGS needs closure
    else:
        raise ValueError(f"Unsupported PyTorch optimizer: {optimizer_name}")

    pinn_model.eval()
    history = {'loss': []}
    start_time = time.time()

    # --- Regularization ---
    reg_weight = 0.0
    if opt_config:
         reg_weight = opt_config.get('parameter_regularization_weight', 0.0)

    def closure():
        optimizer.zero_grad()
        # 1. Interpolate current uplift parameters (differentiable)
        # Assuming interpolation_fn takes tensor and returns tensor
        uplift_grid_torch = interpolation_fn(initial_uplift_params, initial_uplift_params.shape, target_dem_shape) # Adjust args as needed

        # 2. Prepare inputs
        current_params = {
            **{k: v for k, v in model_inputs_template.items() if k != 'initial_topography' and k != 't_target'}, # Get K, D etc.
            'U': uplift_grid_torch
        }
        model_input_tuple = (model_inputs_template['initial_topography'], current_params, model_inputs_template['t_target'])

        # 3. Run PINN prediction
        pred_topo_torch = pinn_model(x=model_input_tuple, mode='predict_state')

        # 4. Calculate similarity loss (differentiable)
        loss = similarity_fn(pred_topo_torch, target_dem)

        # 5. Add regularization
        if reg_weight > 0:
             loss += reg_weight * torch.mean(initial_uplift_params**2)

        # Backward pass
        loss.backward()
        return loss

    # Optimization loop
    for i in tqdm(range(max_iter), desc="Optimizing (Torch)"):
        if optimizer_name.lower() == 'lbfgs':
            loss = optimizer.step(closure)
        else:
            optimizer.zero_grad()
            # 1. Interpolate
            uplift_grid_torch = interpolation_fn(initial_uplift_params, initial_uplift_params.shape, target_dem_shape)
            # 2. Prepare inputs
            current_params = {
                 **{k: v for k, v in model_inputs_template.items() if k != 'initial_topography' and k != 't_target'},
                 'U': uplift_grid_torch
            }
            model_input_tuple = (model_inputs_template['initial_topography'], current_params, model_inputs_template['t_target'])
            # 3. Predict
            pred_topo_torch = pinn_model(x=model_input_tuple, mode='predict_state')
            # 4. Loss
            loss = similarity_fn(pred_topo_torch, target_dem)
            # 5. Regularization
            if reg_weight > 0:
                 loss += reg_weight * torch.mean(initial_uplift_params**2)
            # Backward
            loss.backward()
            optimizer.step()

        # Apply bounds if provided (simple projection)
        if bounds is not None:
             with torch.no_grad():
                  initial_uplift_params.clamp_(min=bounds[0], max=bounds[1])

        current_loss = loss.item()
        history['loss'].append(current_loss)
        if i % 10 == 0:
             logging.info(f"Iter {i}, Loss: {current_loss:.6f}")

    end_time = time.time()
    logging.info(f"PyTorch optimization finished in {end_time - start_time:.2f} seconds.")

    return initial_uplift_params.detach(), history


# --- Main test block for New PyTorch Optimizer ---
if __name__ == '__main__':
    # Example usage for the new PyTorch-based optimization
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    print("Testing PyTorch-based Parameter Optimizer...")

    # --- Dummy Setup ---
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

        # Dummy Model (Needs predict_state)
        class OptimizerTestDummyModel(torch.nn.Module):
            def __init__(self, H=16, W=16):
                super().__init__()
                self.H, self.W = H, W
                # Dummy conv layer to simulate some processing
                self.conv = torch.nn.Conv2d(1 + 1, 1, kernel_size=3, padding=1) # Input: initial_topo + U

            def forward(self, x, mode):
                if mode == 'predict_state':
                    initial_topo = x['initial_state'] # [B, 1, H, W]
                    params = x['params']
                    t_target = x['t_target'] # Scalar or [B] or [B,1,1,1]
                    U_grid = params.get('U') # [B, 1, H, W]

                    if U_grid is None: U_grid = torch.zeros_like(initial_topo)

                    # Ensure t_target can broadcast
                    if isinstance(t_target, torch.Tensor) and t_target.ndim == 1:
                         t_target = t_target.view(-1, 1, 1, 1)

                    # Simple mock prediction: initial + conv(initial, U) * t
                    combined_input = torch.cat([initial_topo, U_grid], dim=1) # [B, 2, H, W]
                    processed = self.conv(combined_input) # [B, 1, H, W]
                    pred = initial_topo + processed * t_target * 0.1 # Scaled effect
                    return pred
                elif mode == 'predict_coords':
                     # Dummy implementation for predict_coords if needed by other parts
                     coords = x
                     return torch.zeros_like(coords['x']) # Return zeros
                return None
        dummy_model = OptimizerTestDummyModel(H=16, W=16).to(device)

        # Dummy Target Data
        H, W = 16, 16
        true_uplift_np = np.random.rand(H, W) * 0.002 # Spatially variable true uplift
        true_uplift_torch = torch.from_numpy(true_uplift_np).float().unsqueeze(0).unsqueeze(0).to(device)
        initial_topo_torch = torch.rand(1, 1, H, W, device=device) * 10 # Random initial topo
        t_target_val = 1000.0
        true_params = {'U': true_uplift_torch, 'K': 1e-5, 'D': 0.01} # Fixed K, D
        with torch.no_grad():
             dummy_target = dummy_model(x={'initial_state': initial_topo_torch, 'params': true_params, 't_target': t_target_val}, mode='predict_state')
        logging.info(f"Generated dummy target data with shape: {dummy_target.shape}")

        # Parameters to Optimize Config
        params_to_opt_config = {
            'U': { # Optimize the 'U' parameter
                'initial_value': torch.zeros_like(true_uplift_torch) + 0.0005, # Initial guess (uniform)
                'bounds': (0.0, 0.005) # Example bounds for uplift
            }
            # Add other parameters here if needed, e.g., 'K'
            # 'K': {'initial_value': 5e-6, 'bounds': (1e-7, 1e-4)}
        }

        # Main Config for Optimization
        dummy_main_config = {
            'optimization_params': {
                'optimizer': 'AdamW',
                'learning_rate': 5e-4, # Adjusted LR
                'max_iterations': 200, # Fewer iterations for test
                'spatial_smoothness_weight': 1e-1, # Add some smoothness penalty
                'log_interval': 20,
                'weight_decay': 1e-4, # For AdamW
                'save_path': 'results/dummy_optimize_test/optimized_params.pth' # Example save path
            }
            # Add other sections like 'physics_params' if needed by objective/model
        }

        # --- Run Optimization ---
        print("\nRunning dummy optimization with PyTorch...")
        optimized_params, history = optimize_parameters(
            model=dummy_model,
            observation_data=dummy_target,
            params_to_optimize_config=params_to_opt_config,
            config=dummy_main_config,
            initial_state=initial_topo_torch,
            fixed_params={'K': true_params['K'], 'D': true_params['D']}, # Pass fixed params
            t_target=t_target_val
        )
        print("Optimization finished.")

        # --- Analyze Results ---
        if optimized_params and 'U' in optimized_params:
            optimized_U = optimized_params['U']
            initial_U = params_to_opt_config['U']['initial_value']
            print(f"Initial U mean: {initial_U.mean().item():.6f}")
            print(f"Optimized U mean: {optimized_U.mean().item():.6f} (True mean: {true_uplift_torch.mean().item():.6f})")
            print(f"Final Loss: {history['final_loss']:.6e}")

            # Optional: Visualize or compare optimized vs true U
            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            # im0 = axes[0].imshow(initial_U.squeeze().cpu().numpy())
            # axes[0].set_title("Initial U")
            # plt.colorbar(im0, ax=axes[0])
            # im1 = axes[1].imshow(optimized_U.squeeze().cpu().numpy())
            # axes[1].set_title("Optimized U")
            # plt.colorbar(im1, ax=axes[1])
            # im2 = axes[2].imshow(true_uplift_torch.squeeze().cpu().numpy())
            # axes[2].set_title("True U")
            # plt.colorbar(im2, ax=axes[2])
            # plt.tight_layout()
            # plt.show()
        else:
            print("Optimization did not return expected parameters.")

    except ImportError as e:
         print(f"Skipping optimizer test due to missing dependency: {e}")
    except Exception as e:
        print(f"Error during optimizer test: {e}")
        import traceback
        traceback.print_exc()


    print("\nOptimizer utilities testing done.")


# def run_optimization(objective_fn, closure, initial_params_dict, config):
#     """
#     (Commented out - Replaced by the new optimize_parameters function)
#     Runs the optimization loop.
#     ... (original docstring) ...
#     """
#     opt_params = config.get('optimization_params', {})
#     optimizer_name = opt_params.get('optimizer', 'LBFGS').lower()
#     lr = opt_params.get('learning_rate', 0.1)
#     max_iterations = opt_params.get('max_iterations', 100)
#
#     params_list = list(initial_params_dict.values())
#     for p in params_list:
#         if not p.requires_grad:
#             logging.warning(f"Parameter {p} does not require grad. Setting requires_grad=True.")
#             p.requires_grad_(True)
#
#     # Setup optimizer (LBFGS, Adam, AdamW)
#     if optimizer_name == 'lbfgs':
#         optimizer = optim.LBFGS(
#             params_list, lr=lr, max_iter=opt_params.get('lbfgs_max_iter', 20),
#             tolerance_grad=opt_params.get('tolerance_grad', 1e-7),
#             tolerance_change=opt_params.get('tolerance_change', 1e-9),
#             history_size=opt_params.get('history_size', 10), line_search_fn="strong_wolfe"
#         )
#         if closure is None: raise ValueError("LBFGS optimizer requires a closure function.")
#     elif optimizer_name == 'adam':
#         optimizer = optim.Adam(params_list, lr=lr, betas=opt_params.get('betas', [0.9, 0.999]), eps=opt_params.get('eps', 1e-8))
#     elif optimizer_name == 'adamw':
#         optimizer = optim.AdamW(params_list, lr=lr, betas=opt_params.get('betas', [0.9, 0.999]), eps=opt_params.get('eps', 1e-8), weight_decay=opt_params.get('weight_decay', 0))
#     else: raise ValueError(f"Unsupported optimizer: {optimizer_name}")
#
#     logging.info(f"Starting optimization with {optimizer_name} for {max_iterations} iterations.")
#     history = {'loss': [], 'iterations': 0, 'time': 0.0}
#     start_time = time.time()
#
#     # Optimization loop
#     for i in tqdm(range(max_iterations), desc="Optimizing"):
#         try:
#             if optimizer_name == 'lbfgs':
#                 loss = optimizer.step(closure)
#             else:
#                 optimizer.zero_grad()
#                 loss, loss_dict = objective_fn(initial_params_dict)
#                 if torch.is_tensor(loss) and loss.requires_grad: loss.backward()
#                 elif not torch.is_tensor(loss):
#                      logging.error(f"Iteration {i}: Objective function did not return a tensor.")
#                      break
#                 optimizer.step()
#
#             current_loss = loss.item()
#             history['loss'].append(current_loss)
#
#             if i % opt_params.get('log_interval', 10) == 0 or i == max_iterations - 1:
#                 logging.info(f"Iteration {i}/{max_iterations}, Loss: {current_loss:.6e}")
#
#             # Convergence checks (Adam/AdamW)
#             if optimizer_name != 'lbfgs':
#                  if i > 0 and abs(history['loss'][-1] - history['loss'][-2]) < opt_params.get('loss_tolerance', 1e-9):
#                      logging.info(f"Converged at iteration {i} due to small loss change.")
#                      break
#                  if not np.isfinite(current_loss):
#                       logging.error(f"Loss is NaN or Inf at iteration {i}. Stopping optimization.")
#                       break
#         except Exception as e:
#             logging.exception(f"Error during optimization iteration {i}:")
#             break
#
#     end_time = time.time()
#     total_time = end_time - start_time
#     history['iterations'] = len(history['loss'])
#     history['time'] = total_time
#     history['final_loss'] = history['loss'][-1] if history['loss'] else float('nan')
#     logging.info(f"Optimization took {total_time:.2f} seconds for {history['iterations']} iterations.")
#     optimized_detached = {k: v.detach() for k, v in initial_params_dict.items()}
#     return optimized_detached, history

# (Keep the __main__ block from the previous version for testing the new functions)
if __name__ == '__main__':
    # Example usage for the new PyTorch-based optimization
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    print("Testing PyTorch-based Parameter Optimizer...")

    # --- Dummy Setup ---
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

        # Dummy Model (Needs predict_state)
        class OptimizerTestDummyModel(torch.nn.Module):
            def __init__(self, H=16, W=16):
                super().__init__()
                self.H, self.W = H, W
                # Dummy conv layer to simulate some processing
                self.conv = torch.nn.Conv2d(1 + 1, 1, kernel_size=3, padding=1) # Input: initial_topo + U

            def forward(self, x, mode):
                if mode == 'predict_state':
                    initial_topo = x['initial_state'] # [B, 1, H, W]
                    params = x['params']
                    t_target = x['t_target'] # Scalar or [B] or [B,1,1,1]
                    U_grid = params.get('U') # [B, 1, H, W]

                    if U_grid is None: U_grid = torch.zeros_like(initial_topo)

                    # Ensure t_target can broadcast
                    if isinstance(t_target, torch.Tensor) and t_target.ndim == 1:
                         t_target = t_target.view(-1, 1, 1, 1)

                    # Simple mock prediction: initial + conv(initial, U) * t
                    combined_input = torch.cat([initial_topo, U_grid], dim=1) # [B, 2, H, W]
                    processed = self.conv(combined_input) # [B, 1, H, W]
                    pred = initial_topo + processed * t_target * 0.1 # Scaled effect
                    return pred
                elif mode == 'predict_coords':
                     # Dummy implementation for predict_coords if needed by other parts
                     coords = x
                     return torch.zeros_like(coords['x']) # Return zeros
                return None
        dummy_model = OptimizerTestDummyModel(H=16, W=16).to(device)

        # Dummy Target Data
        H, W = 16, 16
        true_uplift_np = np.random.rand(H, W) * 0.002 # Spatially variable true uplift
        true_uplift_torch = torch.from_numpy(true_uplift_np).float().unsqueeze(0).unsqueeze(0).to(device)
        initial_topo_torch = torch.rand(1, 1, H, W, device=device) * 10 # Random initial topo
        t_target_val = 1000.0
        true_params = {'U': true_uplift_torch, 'K': 1e-5, 'D': 0.01} # Fixed K, D
        with torch.no_grad():
             dummy_target = dummy_model(x={'initial_state': initial_topo_torch, 'params': true_params, 't_target': t_target_val}, mode='predict_state')
        logging.info(f"Generated dummy target data with shape: {dummy_target.shape}")

        # Parameters to Optimize Config
        params_to_opt_config = {
            'U': { # Optimize the 'U' parameter
                'initial_value': torch.zeros_like(true_uplift_torch) + 0.0005, # Initial guess (uniform)
                'bounds': (0.0, 0.005) # Example bounds for uplift
            }
            # Add other parameters here if needed, e.g., 'K'
            # 'K': {'initial_value': 5e-6, 'bounds': (1e-7, 1e-4)}
        }

        # Main Config for Optimization
        dummy_main_config = {
            'optimization_params': {
                'optimizer': 'AdamW',
                'learning_rate': 5e-4, # Adjusted LR
                'max_iterations': 200, # Fewer iterations for test
                'spatial_smoothness_weight': 1e-1, # Add some smoothness penalty
                'log_interval': 20,
                'weight_decay': 1e-4, # For AdamW
                'save_path': 'results/dummy_optimize_test/optimized_params.pth' # Example save path
            }
            # Add other sections like 'physics_params' if needed by objective/model
        }

        # --- Run Optimization ---
        print("\nRunning dummy optimization with PyTorch...")
        optimized_params, history = optimize_parameters(
            model=dummy_model,
            observation_data=dummy_target,
            params_to_optimize_config=params_to_opt_config,
            config=dummy_main_config,
            initial_state=initial_topo_torch,
            fixed_params={'K': true_params['K'], 'D': true_params['D']}, # Pass fixed params
            t_target=t_target_val
        )
        print("Optimization finished.")

        # --- Analyze Results ---
        if optimized_params and 'U' in optimized_params:
            optimized_U = optimized_params['U']
            initial_U = params_to_opt_config['U']['initial_value']
            print(f"Initial U mean: {initial_U.mean().item():.6f}")
            print(f"Optimized U mean: {optimized_U.mean().item():.6f} (True mean: {true_uplift_torch.mean().item():.6f})")
            print(f"Final Loss: {history['final_loss']:.6e}")

            # Optional: Visualize or compare optimized vs true U
            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            # im0 = axes[0].imshow(initial_U.squeeze().cpu().numpy())
            # axes[0].set_title("Initial U")
            # plt.colorbar(im0, ax=axes[0])
            # im1 = axes[1].imshow(optimized_U.squeeze().cpu().numpy())
            # axes[1].set_title("Optimized U")
            # plt.colorbar(im1, ax=axes[1])
            # im2 = axes[2].imshow(true_uplift_torch.squeeze().cpu().numpy())
            # axes[2].set_title("True U")
            # plt.colorbar(im2, ax=axes[2])
            # plt.tight_layout()
            # plt.show()
        else:
            print("Optimization did not return expected parameters.")

    except ImportError as e:
         print(f"Skipping optimizer test due to missing dependency: {e}")
    except Exception as e:
        print(f"Error during optimizer test: {e}")
        import traceback
        traceback.print_exc()


    print("\nOptimizer utilities testing done.")
    print("\nOptimizer utilities testing done.")

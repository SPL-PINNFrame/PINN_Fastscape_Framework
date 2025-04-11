import torch
import torch.optim as optim
import logging
from tqdm import tqdm
import time
import numpy as np
import torch.nn.functional as F
from src.physics import calculate_laplacian
from scipy.interpolate import griddata
import os # For save_path

# --- 统一的插值函数 ---

def interpolate_uplift(uplift_params, param_shape, target_shape, backend='torch', 
                       method='rbf', sigma=0.1, **kwargs):
    """统一的插值函数接口
    
    Args:
        uplift_params: 参数数组/张量
        param_shape: 参数原始形状
        target_shape: 目标形状
        backend: 使用哪个后端 ('torch' 或 'numpy')
        method: 插值方法 (torch: 'rbf', 'bilinear'; numpy: 'linear', 'nearest', 'cubic')
        sigma: RBF带宽参数 (用于torch+rbf方法)
        **kwargs: 其他参数
        
    Returns:
        插值后的数组/张量
    """
    if backend == 'torch':
        # 使用PyTorch实现
        # 确保输入是张量
        from .utils import ensure_tensor
        uplift_params = ensure_tensor(uplift_params)
        device = uplift_params.device
        dtype = uplift_params.dtype
        
        # 确保输入是扁平化的
        if uplift_params.ndim != 1:
            uplift_params_flat = uplift_params.flatten()
        else:
            uplift_params_flat = uplift_params
            
        param_h, param_w = param_shape
        target_h, target_w = target_shape
        
        # 创建归一化的源网格坐标
        x_src = torch.linspace(0, 1, param_w, device=device, dtype=dtype)
        y_src = torch.linspace(0, 1, param_h, device=device, dtype=dtype)
        grid_y_src, grid_x_src = torch.meshgrid(y_src, x_src, indexing='ij')
        points_src = torch.stack([grid_x_src.flatten(), grid_y_src.flatten()], dim=1) # Shape (H_param*W_param, 2)
        
        # 创建归一化的目标网格坐标
        x_tgt = torch.linspace(0, 1, target_w, device=device, dtype=dtype)
        y_tgt = torch.linspace(0, 1, target_h, device=device, dtype=dtype)
        grid_y_tgt, grid_x_tgt = torch.meshgrid(y_tgt, x_tgt, indexing='ij')
        points_tgt = torch.stack([grid_x_tgt.flatten(), grid_y_tgt.flatten()], dim=1) # Shape (H_target*W_target, 2)
        
        # 实现不同的插值方法
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
            values_tgt = torch.matmul(weights, uplift_params_flat.unsqueeze(1)).squeeze(1) # [Q, P] @ [P, 1] -> [Q, 1] -> [Q]
            return values_tgt.reshape(target_shape)
            
        elif method == 'bilinear':
            # 使用grid_sample实现双线性插值
            from .utils import normalize_grid_sample
            # 需要将目标坐标从[0,1]映射到[-1,1] for grid_sample
            grid_sample_coords = torch.stack([grid_x_tgt, grid_y_tgt], dim=2).unsqueeze(0) # Shape (1, H_target, W_target, 2)
            grid_sample_coords = normalize_grid_sample(grid_sample_coords)
            
            # 重塑参数网格并添加批次和通道维度
            param_grid = uplift_params_flat.reshape(1, 1, param_h, param_w) # Shape (1, 1, H_param, W_param)
            
            # 使用grid_sample进行插值
            # align_corners=True is often recommended for resolution changes
            values_tgt_grid = F.grid_sample(
                param_grid,
                grid_sample_coords,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )
            return values_tgt_grid.squeeze() # Remove B, C dims -> Shape (H_target, W_target)
        else:
            raise ValueError(f"未知的PyTorch插值方法: {method}")
        
    elif backend == 'numpy':
        # 使用SciPy实现
        if isinstance(uplift_params, torch.Tensor):
            uplift_params_np = uplift_params.detach().cpu().numpy()
        else:
            uplift_params_np = uplift_params
            
        if uplift_params_np.ndim != 1:
            uplift_params_flat = uplift_params_np.flatten()
        else:
            uplift_params_flat = uplift_params_np
            
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
            interpolated_uplift_flat = griddata(param_points, uplift_params_flat, target_points, 
                                               method=method, fill_value=np.mean(uplift_params_flat))
            # Handle potential NaNs if method='cubic'
            interpolated_uplift_flat = np.nan_to_num(interpolated_uplift_flat, nan=np.mean(uplift_params_flat))
        except Exception as e:
            logging.error(f"Interpolation failed: {e}. Returning mean value grid.")
            interpolated_uplift_flat = np.full(target_points.shape[0], np.mean(uplift_params_flat))

        return interpolated_uplift_flat.reshape(target_shape)
        
    else:
        raise ValueError(f"未知的后端: {backend}")

# 保留向后兼容的函数名
def interpolate_uplift_cv(uplift_params_flat, param_shape, target_shape, method='linear'):
    """保留向后兼容的SciPy版本插值"""
    return interpolate_uplift(uplift_params_flat, param_shape, target_shape, 
                              backend='numpy', method=method)

def interpolate_uplift_torch(uplift_params, param_shape, target_shape, method='rbf', sigma=0.1):
    """保留向后兼容的PyTorch版本插值"""
    return interpolate_uplift(uplift_params, param_shape, target_shape, 
                              backend='torch', method=method, sigma=sigma)

# --- Parameter Optimizer Class ---

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
        from .utils import prepare_parameter
        return prepare_parameter(
            initial_value, 
            target_shape=(self.height, self.width), 
            batch_size=self.batch_size, 
            device=self.device, 
            dtype=self.dtype, 
            param_name=param_name
        )


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


# --- PyTorch-based Optimization Function ---

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

# --- PyTorch Optimization Wrapper ---

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
    运行使用PyTorch优化器的uplift优化。
    需要可微分的插值和相似度函数。
    
    用optimize_parameters函数替代此函数，以获得更完整的功能和更好的错误处理。
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


# --- 示例测试代码 ---
if __name__ == '__main__':
    # 示例用法
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    print("Testing PyTorch-based Parameter Optimizer...")

    # --- 测试设置 ---
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

        # 模拟模型
        class OptimizerTestDummyModel(torch.nn.Module):
            def __init__(self, H=16, W=16):
                super().__init__()
                self.H, self.W = H, W
                # 模拟处理层
                self.conv = torch.nn.Conv2d(1 + 1, 1, kernel_size=3, padding=1) # 输入: initial_topo + U

            def forward(self, x, mode):
                if mode == 'predict_state':
                    initial_topo = x['initial_state'] # [B, 1, H, W]
                    params = x['params']
                    t_target = x['t_target'] # 标量或 [B] 或 [B,1,1,1]
                    U_grid = params.get('U') # [B, 1, H, W]

                    if U_grid is None: U_grid = torch.zeros_like(initial_topo)

                    # 确保t_target可以广播
                    if isinstance(t_target, torch.Tensor) and t_target.ndim == 1:
                         t_target = t_target.view(-1, 1, 1, 1)

                    # 简单的模拟预测: initial + conv(initial, U) * t
                    combined_input = torch.cat([initial_topo, U_grid], dim=1) # [B, 2, H, W]
                    processed = self.conv(combined_input) # [B, 1, H, W]
                    pred = initial_topo + processed * t_target * 0.1 # 缩放效果
                    return pred
                elif mode == 'predict_coords':
                     # predict_coords的虚拟实现
                     coords = x
                     return torch.zeros_like(coords['x']) # 返回零
                return None
        dummy_model = OptimizerTestDummyModel(H=16, W=16).to(device)

        # 虚拟目标数据
        H, W = 16, 16
        true_uplift_np = np.random.rand(H, W) * 0.002 # 空间变化的真实uplift
        true_uplift_torch = torch.from_numpy(true_uplift_np).float().unsqueeze(0).unsqueeze(0).to(device)
        initial_topo_torch = torch.rand(1, 1, H, W, device=device) * 10 # 随机初始地形
        t_target_val = 1000.0
        true_params = {'U': true_uplift_torch, 'K': 1e-5, 'D': 0.01} # 固定 K, D
        with torch.no_grad():
             dummy_target = dummy_model(x={'initial_state': initial_topo_torch, 'params': true_params, 't_target': t_target_val}, mode='predict_state')
        logging.info(f"Generated dummy target data with shape: {dummy_target.shape}")

        # 参数优化配置
        params_to_opt_config = {
            'U': { # 优化'U'参数
                'initial_value': torch.zeros_like(true_uplift_torch) + 0.0005, # 初始猜测（均匀）
                'bounds': (0.0, 0.005) # uplift的范围示例
            }
            # 如果需要可以添加其他参数，例如'K'
            # 'K': {'initial_value': 5e-6, 'bounds': (1e-7, 1e-4)}
        }

        # 主优化配置
        dummy_main_config = {
            'optimization_params': {
                'optimizer': 'AdamW',
                'learning_rate': 5e-4, # 调整学习率
                'max_iterations': 200, # 减少测试的迭代次数
                'spatial_smoothness_weight': 1e-1, # 添加平滑度惩罚
                'log_interval': 20,
                'weight_decay': 1e-4, # 用于AdamW
                'save_path': 'results/dummy_optimize_test/optimized_params.pth' # 保存路径示例
            }
            # 如果目标/模型需要，可以添加其他部分如'physics_params'
        }

        # --- 运行优化 ---
        print("\n使用PyTorch运行虚拟优化...")
        optimized_params, history = optimize_parameters(
            model=dummy_model,
            observation_data=dummy_target,
            params_to_optimize_config=params_to_opt_config,
            config=dummy_main_config,
            initial_state=initial_topo_torch,
            fixed_params={'K': true_params['K'], 'D': true_params['D']}, # 传递固定参数
            t_target=t_target_val
        )
        print("优化完成.")

        # --- 分析结果 ---
        if optimized_params and 'U' in optimized_params:
            optimized_U = optimized_params['U']
            initial_U = params_to_opt_config['U']['initial_value']
            print(f"Initial U mean: {initial_U.mean().item():.6f}")
            print(f"Optimized U mean: {optimized_U.mean().item():.6f} (True mean: {true_uplift_torch.mean().item():.6f})")
            print(f"Final Loss: {history['final_loss']:.6e}")

            # 可选: 可视化或比较优化的vs真实U
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
            print("优化未返回预期参数。")

    except ImportError as e:
         print(f"由于缺少依赖跳过优化器测试: {e}")
    except Exception as e:
        print(f"优化器测试期间出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n优化器工具测试完成。")
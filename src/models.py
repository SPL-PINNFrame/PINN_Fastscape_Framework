import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
import math # For sqrt(2) in AdaptiveFastscapePINN tiling (if needed)

# --- Improved Base Class for Dual Output --- 
class TimeDerivativePINN(nn.Module):
    """能同时输出状态及其时间导数的PINN基类"""
    
    def __init__(self):
        super().__init__()
        self.output_state = True
        self.output_derivative = True
        # Add output mode tracker for debugging
        self._mode_changes = [] 
    
    def get_output_mode(self):
        """获取当前输出模式"""
        modes = []
        if self.output_state:
            modes.append('state')
        if self.output_derivative:
            modes.append('derivative')
        return modes
    
    def set_output_mode(self, state=True, derivative=True):
        """设置输出模式（状态和/或导数）
        
        Args:
            state (bool): 是否输出状态
            derivative (bool): 是否输出时间导数
            
        Raises:
            ValueError: 如果state和derivative均为False
        """
        if not state and not derivative:
            raise ValueError("至少需要一个输出模式为True（state或derivative）")
        
        # Track mode changes for debugging
        old_modes = self.get_output_mode()
        self.output_state = state
        self.output_derivative = derivative
        new_modes = self.get_output_mode()
        
        # Log mode changes
        if old_modes != new_modes:
            self._mode_changes.append((old_modes, new_modes))
            logging.debug(f"TimeDerivativePINN output mode changed: {old_modes} -> {new_modes}")
    
    def check_output_format(self, outputs, required_outputs=None):
        """检查输出格式是否符合预期
        
        Args:
            outputs: 模型输出（字典或张量）
            required_outputs (list, optional): 需要的输出类型列表，例如 ['state', 'derivative']
                                            
        Returns:
            bool: 输出格式是否符合预期
            
        Raises:
            ValueError: 如果输出格式不符合预期且required_outputs不为None
        """
        if required_outputs is None:
            # 只检查模式配置和输出类型匹配
            if isinstance(outputs, dict):
                if self.output_state and 'state' not in outputs:
                    if required_outputs:
                        raise ValueError("模型配置为输出状态，但输出字典中没有'state'键")
                    return False
                if self.output_derivative and 'derivative' not in outputs:
                    if required_outputs:
                        raise ValueError("模型配置为输出导数，但输出字典中没有'derivative'键")
                    return False
                return True
            else:
                # 单一输出应该是state（如果只配置了state模式）
                if self.output_state and not self.output_derivative:
                    return True
                # 否则，应该是字典格式
                if required_outputs:
                    raise ValueError(f"模型配置为输出 {self.get_output_mode()}，但返回了单一张量而非字典")
                return False
        else:
            # 检查是否有所有需要的输出
            if isinstance(outputs, dict):
                for output_type in required_outputs:
                    if output_type not in outputs:
                        if required_outputs:
                            raise ValueError(f"需要的输出'{output_type}'不在模型输出字典中")
                        return False
                return True
            else:
                # 单一输出只能满足单一需求
                if len(required_outputs) == 1 and required_outputs[0] == 'state':
                    return True
                if required_outputs:
                    raise ValueError(f"需要输出类型 {required_outputs}，但模型返回了单一张量")
                return False
    
    def forward(self, *args, **kwargs):
        """前向传播，需要在子类中实现"""
        raise NotImplementedError("子类必须实现forward方法")
    
    def predict_derivative_fd(self, x, delta_t=1e-3, mode='predict_coords'):
        """使用有限差分近似计算时间导数（用于测试）
        
        Args:
            x: 输入数据（取决于mode）
            delta_t (float): 时间步长
            mode (str): 预测模式，与forward相同
            
        Returns:
            torch.Tensor: 时间导数近似值
        """
        # 保存当前输出模式
        original_state = self.output_state
        original_derivative = self.output_derivative
        
        # 只输出状态用于有限差分
        self.set_output_mode(state=True, derivative=False)
        
        # 创建前向和后向时间输入副本
        if mode == 'predict_coords':
            # 坐标模式下，修改't'键
            if not isinstance(x, dict) or 't' not in x:
                raise ValueError("predict_derivative_fd在'predict_coords'模式下需要't'键")
            
            t = x['t']
            
            # 前向时间步
            x_forward = x.copy()
            x_forward['t'] = t + delta_t/2
            
            # 后向时间步
            x_backward = x.copy()
            x_backward['t'] = t - delta_t/2
            
        elif mode == 'predict_state':
            # 状态模式下，修改't_target'键
            if not isinstance(x, dict) or 't_target' not in x:
                raise ValueError("predict_derivative_fd在'predict_state'模式下需要't_target'键")
            
            t_target = x['t_target']
            
            # 前向时间步
            x_forward = x.copy()
            x_forward['t_target'] = t_target + delta_t/2
            
            # 后向时间步
            x_backward = x.copy()
            x_backward['t_target'] = t_target - delta_t/2
        
        else:
            raise ValueError(f"predict_derivative_fd不支持的模式: {mode}")
        
        # 计算前向和后向预测
        with torch.no_grad():  # 防止创建计算图，这只是用于测试
            pred_forward = self.forward(x_forward, mode=mode)
            pred_backward = self.forward(x_backward, mode=mode)
        
        # 使用中心差分计算导数
        if isinstance(pred_forward, dict) and 'state' in pred_forward:
            derivative_fd = (pred_forward['state'] - pred_backward['state']) / delta_t
        else:
            derivative_fd = (pred_forward - pred_backward) / delta_t
        
        # 恢复原始模式
        self.set_output_mode(state=original_state, derivative=original_derivative)
        
        return derivative_fd

# --- MLP_PINN ---
class MLP_PINN(nn.Module):
    """
    简单的多层感知机 (MLP) PINN。
    输入坐标 (x, y, t) 或 (x, y, t, param1, param2, ...)，预测地形 h。
    也支持基于网格的状态预测。
    """
    # REMOVED dtype=torch.float32 from signature
    def __init__(self, input_dim=3, output_dim=1, hidden_layers=8, hidden_neurons=256, activation=nn.Tanh()):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim # Store output_dim

        layers = []
        # REMOVED dtype=dtype from nn.Linear
        layers.append(nn.Linear(input_dim, hidden_neurons))
        layers.append(activation)
        for _ in range(hidden_layers - 1):
            # REMOVED dtype=dtype from nn.Linear
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(activation)
        # REMOVED dtype=dtype from nn.Linear
        layers.append(nn.Linear(hidden_neurons, output_dim))
        self.network = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        """初始化网络权重 (Xavier Uniform)。"""
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                gain = 1.0
                if len(self.network) > 1 and hasattr(self.network[1], '__class__') and hasattr(nn.init, 'calculate_gain'):
                     try:
                          # Get activation function name (handle Sequential case)
                          activation_module = self.network[1]
                          if isinstance(activation_module, nn.Sequential):
                               activation_func = activation_module[0] # Assume first layer in seq is activation
                          else:
                               activation_func = activation_module
                          gain = nn.init.calculate_gain(activation_func.__class__.__name__.lower())
                     except ValueError:
                          gain = 1.0 # Default gain if activation not recognized
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _prepare_coord_input(self, coords):
        """准备并验证坐标输入以进行预测。"""
        expected_keys = []
        if self.input_dim >= 1: expected_keys.append('x')
        if self.input_dim >= 2: expected_keys.append('y')
        if self.input_dim >= 3: expected_keys.append('t')
        # 假设额外的维度是 k, u
        if self.input_dim == 5:
            expected_keys.extend(['k', 'u'])
        elif self.input_dim > 3:
            # 为其他参数添加占位符键
            for i in range(3, self.input_dim):
                 expected_keys.append(f'param{i-2}')
            logging.warning(f"MLP_PINN input_dim={self.input_dim} > 3. Assuming extra inputs are 'param1', 'param2', ...")

        tensors_to_cat = []
        for key in expected_keys:
            if key not in coords:
                # 如果是可选参数（k, u, paramX）且未提供，则使用零填充
                if key in ['k', 'u'] or key.startswith('param'):
                     # 需要知道 x 的形状来创建零张量
                     ref_shape = coords.get('x', coords.get('y', coords.get('t'))).shape
                     if ref_shape is None:
                          raise ValueError(f"无法确定形状以创建零填充参数 '{key}'")
                     logging.debug(f"Parameter '{key}' not found in coords, using zeros.")
                     tensors_to_cat.append(torch.zeros(ref_shape, device=coords[next(iter(coords))].device, dtype=coords[next(iter(coords))].dtype))
                else:
                     raise ValueError(f"缺少必需的坐标键 '{key}' (input_dim={self.input_dim})")
            else:
                tensors_to_cat.append(coords[key])

        if not tensors_to_cat:
             raise ValueError("未找到用于 MLP 输入的张量。")

        model_input = torch.cat(tensors_to_cat, dim=-1) # Concatenate along the last dimension

        # 验证最终输入形状
        if model_input.shape[-1] != self.input_dim:
             raise ValueError(f"MLP 输入维度不匹配。预期 {self.input_dim}, 得到 {model_input.shape[-1]}")

        return model_input

    def forward(self, x, mode='predict_coords'):
        """
        前向传播，适应不同预测模式。

        Args:
            x: 输入数据。
               - mode='predict_coords': x 是包含坐标张量的字典，例如 {'x': [N,1], 'y': [N,1], 't': [N,1], 'k': [N,1], ...}
                                        或 {'x': [B,N,1], 'y': [B,N,1], ...}
               - mode='predict_state': x 是包含初始状态、参数和目标时间的字典，
                                       例如 {'initial_state': [B,1,H,W], 'params': {'K':..., 'U':...}, 't_target': ...}
            mode (str): 'predict_coords' 或 'predict_state'。

        Returns:
            torch.Tensor: 预测的地形。形状取决于模式。
        """
        if mode == 'predict_coords':
            if not isinstance(x, dict):
                 raise TypeError("对于 'predict_coords' 模式，输入 x 必须是字典。")
            model_input = self._prepare_coord_input(x)
            h_pred = self.network(model_input)
            return h_pred # Shape [N, C_out] or [B, N, C_out]

        elif mode == 'predict_state':
            # 实现基于网格的状态预测 (方案 2)
            if not isinstance(x, dict):
                 raise TypeError("对于 'predict_state' 模式，输入 x 必须是字典。")

            initial_state = x.get('initial_state') # 可能未使用，但用于获取形状
            params = x.get('params', {})
            t_target = x.get('t_target')

            if initial_state is None or t_target is None:
                 raise ValueError("对于 'predict_state' 模式，缺少 'initial_state' 或 't_target' 输入。")

            batch_size, _, height, width = initial_state.shape
            device = initial_state.device
            dtype = initial_state.dtype # Use dtype from input state

            # 创建网格坐标 [0, 1]
            y_coords_norm = torch.linspace(0, 1, height, device=device, dtype=dtype)
            x_coords_norm = torch.linspace(0, 1, width, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(y_coords_norm, x_coords_norm, indexing='ij') # H, W

            # 展平并添加批次维度: [B, H*W, 1]
            x_flat = grid_x.reshape(1, -1, 1).expand(batch_size, -1, -1)
            y_flat = grid_y.reshape(1, -1, 1).expand(batch_size, -1, -1)

            # 准备时间张量: [B, H*W, 1]
            if isinstance(t_target, (int, float)):
                t_flat = torch.full((batch_size, height * width, 1), float(t_target), device=device, dtype=dtype)
            elif isinstance(t_target, torch.Tensor):
                 if t_target.numel() == 1:
                      t_flat = t_target.expand(batch_size, height * width, 1)
                 elif t_target.ndim == 1 and t_target.shape[0] == batch_size: # [B]
                      t_flat = t_target.view(batch_size, 1, 1).expand(-1, height * width, -1)
                 elif t_target.shape == (batch_size, 1): # [B, 1]
                      t_flat = t_target.unsqueeze(1).expand(-1, height * width, -1)
                 else:
                      raise ValueError(f"不支持的目标时间形状: {t_target.shape}")
            else:
                 raise TypeError(f"不支持的目标时间类型: {type(t_target)}")

            # 准备坐标字典
            coords = {'x': x_flat, 'y': y_flat, 't': t_flat}

            # 添加参数（如果模型需要）
            if self.input_dim > 3:
                param_keys_map = {'k': 'K', 'u': 'U'} # Map internal keys to expected param dict keys
                expected_param_keys = []
                if self.input_dim == 5: expected_param_keys = ['k', 'u']
                elif self.input_dim > 3: expected_param_keys = [f'param{i-2}' for i in range(3, self.input_dim)]

                for key in expected_param_keys:
                    param_name = param_keys_map.get(key, key) # Get K, U etc.
                    param_value = params.get(param_name)

                    if param_value is None:
                         coords[key] = torch.zeros_like(x_flat)
                         logging.debug(f"Parameter '{param_name}' (for key '{key}') not found in params dict, using zeros.")
                         continue

                    # 处理标量或空间变化的参数
                    if isinstance(param_value, (int, float)):
                         param_tensor = torch.full_like(x_flat, float(param_value))
                    elif isinstance(param_value, torch.Tensor):
                         param_value = param_value.to(device=device, dtype=dtype)
                         if param_value.ndim == 0: # Scalar tensor
                              param_tensor = param_value.expand_as(x_flat)
                         elif param_value.ndim == 1 and param_value.shape[0] == batch_size: # Batch scalar [B]
                              param_tensor = param_value.view(batch_size, 1, 1).expand_as(x_flat)
                         elif param_value.ndim >= 2: # Spatial field [B,1,H,W] or [H,W] etc.
                              # Sample from grid if spatial
                              if param_value.ndim == 2: param_value = param_value.unsqueeze(0).unsqueeze(0) # H,W -> 1,1,H,W
                              if param_value.ndim == 3: param_value = param_value.unsqueeze(1) # B,H,W -> B,1,H,W
                              # Use grid_sample for spatial fields
                              grid_for_sample = torch.stack([x_flat.squeeze(-1), y_flat.squeeze(-1)], dim=-1) # [B, H*W, 2]
                              # Convert coords [0,1] to [-1,1] for grid_sample
                              grid_for_sample = 2.0 * grid_for_sample - 1.0
                              # Add dummy dimension for grid_sample: [B, H*W, 1, 2]
                              grid_for_sample = grid_for_sample.unsqueeze(2)
                              sampled_param = F.grid_sample(param_value, grid_for_sample, mode='bilinear', padding_mode='border', align_corners=False)
                              # sampled_param shape: [B, C, N, 1] -> need [B, N, C]
                              param_tensor = sampled_param.squeeze(-1).permute(0, 2, 1) # [B, H*W, C] (assume C=1)
                         else:
                              raise ValueError(f"无法处理参数 '{param_name}' 的形状: {param_value.shape}")
                    else:
                         raise TypeError(f"不支持的参数类型 '{param_name}': {type(param_value)}")

                    coords[key] = param_tensor

            # 使用准备好的坐标调用 predict_coords 模式
            h_pred_flat = self.forward(coords, mode='predict_coords') # Shape [B, H*W, C_out]

            # 重塑回网格形状 [B, C_out, H, W]
            try:
                h_pred_grid = h_pred_flat.permute(0, 2, 1).reshape(batch_size, self.output_dim, height, width)
            except Exception as e:
                 logging.error(f"在 predict_state 中重塑输出时出错: {e}. Flat shape: {h_pred_flat.shape}")
                 raise

            return h_pred_grid

        else:
            raise ValueError(f"未知的 forward 模式: {mode}")

# --- FastscapePINN ---
class FastscapePINN(nn.Module):
    """增强版物理信息神经网络，同时支持点预测和状态预测（使用 CNN 编码器-解码器）"""
    def __init__(self, input_dim=3, output_dim=1, hidden_dim=256, num_layers=8,
                 grid_height=64, grid_width=64, num_param_channels=3):
        super().__init__()
        self.output_dim = output_dim
        self.grid_height = grid_height
        self.grid_width = grid_width

        # 基本MLP网络用于坐标预测
        self.mlp = MLP_PINN(input_dim, output_dim, num_layers, hidden_dim)

        # 状态预测的编码器-解码器架构
        self.num_param_channels = num_param_channels
        encoder_input_channels = 1 + self.num_param_channels # 1 (initial_topo) + params
        self.encoder = nn.Sequential(
            nn.Conv2d(encoder_input_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, output_dim, 3, padding=1)
        )
        self.time_encoder = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())

    def _ensure_param_grid_shape(self, p, key, batch_size, device):
        """辅助函数，确保参数具有正确的网格形状 [B, 1, H, W]。"""
        if p is None:
            logging.warning(f"参数 '{key}' 未在 params 字典中找到，使用 0。")
            return torch.zeros((batch_size, 1, self.grid_height, self.grid_width), device=device, dtype=torch.float32)

        if isinstance(p, (int, float)):
            return torch.full((batch_size, 1, self.grid_height, self.grid_width), float(p), device=device, dtype=torch.float32)
        elif isinstance(p, torch.Tensor):
            p = p.float().to(device)
            if p.ndim == 0: # Scalar tensor
                 return p.view(1, 1, 1, 1).expand(batch_size, 1, self.grid_height, self.grid_width)
            elif p.ndim == 1 and p.shape[0] == batch_size: # Batch of scalars [B]
                return p.view(batch_size, 1, 1, 1).expand(-1, 1, self.grid_height, self.grid_width)
            elif p.ndim == 2 and p.shape == (self.grid_height, self.grid_width): # Spatial field [H, W]
                return p.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
            elif p.ndim == 3 and p.shape[0] == batch_size and p.shape[1:] == (self.grid_height, self.grid_width): # Spatial field [B, H, W]
                return p.unsqueeze(1)
            elif p.ndim == 4 and p.shape == (batch_size, 1, self.grid_height, self.grid_width): # Standard format [B, 1, H, W]
                return p
            else:
                try: # Try broadcasting
                     return p.expand(batch_size, 1, self.grid_height, self.grid_width)
                except RuntimeError:
                     raise ValueError(f"参数 '{key}' 张量形状 {p.shape} 无法广播/处理。")
        else:
             raise TypeError(f"不支持的参数类型 '{key}': {type(p)}")

    def _predict_with_encoder_decoder(self, initial_state, params, t_target):
        """使用编码器-解码器架构进行状态预测。"""
        batch_size = initial_state.shape[0]
        device = initial_state.device

        # 1. 时间编码
        if isinstance(t_target, (int, float)):
            t_tensor = torch.full((batch_size, 1), float(t_target), device=device, dtype=torch.float32)
        elif isinstance(t_target, torch.Tensor):
             t_target = t_target.to(device)
             if t_target.numel() == 1: t_tensor = t_target.expand(batch_size, 1)
             elif t_target.shape == (batch_size, 1) or t_target.shape == (batch_size,): t_tensor = t_target.view(batch_size, 1)
             else: raise ValueError(f"不支持的目标时间张量形状: {t_target.shape}")
        else: raise TypeError(f"不支持的目标时间类型: {type(t_target)}")
        time_features = self.time_encoder(t_tensor.float()) # [B, 64]

        # 2. 参数网格准备
        params_grid = []
        encoder_param_keys = ['K', 'D', 'U'] # Example keys
        if len(encoder_param_keys) != self.num_param_channels:
             logging.warning(f"编码器参数键数量 ({len(encoder_param_keys)}) 与 num_param_channels ({self.num_param_channels}) 不匹配。")
        for key in encoder_param_keys[:self.num_param_channels]: # Use only expected number
            p_grid = self._ensure_param_grid_shape(params.get(key), key, batch_size, device)
            params_grid.append(p_grid)

        # 3. 初始状态准备
        if initial_state.ndim == 3: initial_state = initial_state.unsqueeze(1)
        elif initial_state.ndim != 4 or initial_state.shape[1] != 1: raise ValueError(f"initial_state 形状应为 (B, H, W) 或 (B, 1, H, W)，得到 {initial_state.shape}")
        initial_state = initial_state.float().to(device)

        # 4. 编码器输入
        encoder_input = torch.cat([initial_state] + params_grid, dim=1) # [B, 1 + num_param_channels, H, W]

        # 5. 编码-解码过程
        features = self.encoder(encoder_input)

        # 6. 时间特征融合
        time_features_expanded = time_features.view(batch_size, -1, 1, 1).expand(-1, -1, features.shape[2], features.shape[3])
        num_time_channels = time_features_expanded.shape[1]
        num_feat_channels = features.shape[1]
        fused_features = features.clone()
        channels_to_add = min(num_time_channels, num_feat_channels)
        fused_features[:, :channels_to_add] = features[:, :channels_to_add] + time_features_expanded[:, :channels_to_add]

        # 7. 解码
        prediction = self.decoder(fused_features)
        return prediction # Shape [B, output_dim, H, W]

    def forward(self, x, mode='predict_coords'):
        if mode == 'predict_coords':
            if not isinstance(x, dict): raise TypeError("对于 'predict_coords' 模式，输入 x 必须是字典。")
            return self.mlp(x, mode='predict_coords') # Pass dict directly
        elif mode == 'predict_state':
            if isinstance(x, dict):
                initial_state = x.get('initial_state')
                params = x.get('params')
                t_target = x.get('t_target')
            elif isinstance(x, (tuple, list)) and len(x) == 3:
                initial_state, params, t_target = x
            else:
                raise ValueError("对于 'predict_state' 模式，输入 x 必须是字典或 (initial_state, params, t_target) 元组/列表")
            if initial_state is None or params is None or t_target is None:
                 raise ValueError("对于 'predict_state' 模式，缺少 'initial_state', 'params', 或 't_target' 输入")
            # 调用编码器-解码器实现
            return self._predict_with_encoder_decoder(initial_state, params, t_target)
        else:
            raise ValueError(f"未知的 forward 模式: {mode}")


# --- AdaptiveFastscapePINN (Dual Output) ---
class AdaptiveFastscapePINN(TimeDerivativePINN): # Inherit from base class
    """支持任意尺寸参数矩阵和多分辨率处理的物理信息神经网络"""
    def __init__(self, input_dim=5, output_dim=1, hidden_dim=256, num_layers=8,
                 base_resolution=64, max_resolution=1024, activation=nn.Tanh(),
                 coordinate_input_dim=5, # Add coordinate_input_dim
                 domain_x: list = [0.0, 1.0], # Add domain info with defaults
                 domain_y: list = [0.0, 1.0]):
        super().__init__() # Call TimeDerivativePINN init
        self.output_dim = output_dim
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        # Store domain boundaries
        if not (isinstance(domain_x, (list, tuple)) and len(domain_x) == 2 and isinstance(domain_y, (list, tuple)) and len(domain_y) == 2):
             raise ValueError("domain_x and domain_y must be lists or tuples of length 2.")
        self.domain_x = domain_x
        self.domain_y = domain_y
        self.epsilon = 1e-9 # For safe division during normalization

        # 基础坐标-参数MLP (x, y, t, k, u -> h)
        # Coordinate MLP now needs separate heads
        self.coordinate_mlp_base = MLP_PINN(coordinate_input_dim, hidden_dim, num_layers - 1, hidden_dim, activation=activation) # Output hidden_dim
        self.state_head = nn.Linear(hidden_dim, output_dim)
        self.derivative_head = nn.Linear(hidden_dim, output_dim)

        # CNN 编码器-解码器 (用于网格处理)
        # 输入通道: 1 (地形) + 2 (参数 U, K) = 3
        cnn_input_channels = 3
        self.encoder = nn.Sequential(
            nn.Conv2d(cnn_input_channels, 32, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.LeakyReLU(0.2), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, output_dim, 3, padding=1) # State decoder
        )
        # Add a similar decoder for the derivative output
        self.derivative_decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, output_dim, 3, padding=1)
        )
        # 下采样器，用于多分辨率处理
        self.downsampler = nn.Upsample(size=(base_resolution, base_resolution), mode='bilinear', align_corners=False)

    def _ensure_shape(self, param, target_shape, batch_size, device, dtype):
        """确保参数具有正确的形状，用于网格预测"""
        from .utils import prepare_parameter
        return prepare_parameter(
            param_value=param,
            target_shape=target_shape,
            batch_size=batch_size,
            device=device,
            dtype=dtype
        )

    def _sample_at_coords(self, param_grid, x_coords_norm, y_coords_norm):
        """在参数网格上采样局部值 (使用归一化坐标 [0, 1])"""
        if param_grid is None: return torch.zeros_like(x_coords_norm)
        device = x_coords_norm.device
        dtype = param_grid.dtype # Match param grid dtype

        # 确保 param_grid 是 [B, C, H, W]
        if param_grid.ndim == 2: param_grid = param_grid.unsqueeze(0).unsqueeze(0)
        elif param_grid.ndim == 3: param_grid = param_grid.unsqueeze(1)
        param_grid = param_grid.to(device)

        # 转换坐标到 [-1, 1] for grid_sample
        x_sample = 2.0 * torch.clamp(x_coords_norm, 0, 1) - 1.0
        y_sample = 2.0 * torch.clamp(y_coords_norm, 0, 1) - 1.0

        # 准备采样网格 [B, N, 1, 2]
        grid = torch.stack([x_sample, y_sample], dim=-1) # [N, 1] -> [N, 1, 2]
        if grid.ndim == 3: grid = grid.unsqueeze(0) # Add batch dim if needed -> [1, N, 1, 2]
        if grid.shape[0] != param_grid.shape[0]: grid = grid.expand(param_grid.shape[0], -1, -1, -1)

        # 采样 [B, C, N, 1]
        # MODIFIED: Set align_corners=True for sampling with [0,1] normalized coords converted to [-1,1]
        sampled = F.grid_sample(param_grid, grid.to(dtype), mode='bilinear', padding_mode='border', align_corners=True)

        # Reshape to [N, C] (average over batch if B > 1)
        if sampled.shape[0] > 1: sampled = sampled.mean(dim=0)
        else: sampled = sampled.squeeze(0)
        return sampled.squeeze(-1).permute(1, 0) # [C, N] -> [N, C]

    def _encode_time(self, t_target, batch_size, device, dtype):
        """将时间编码为特征向量 (简化)"""
        if isinstance(t_target, (int, float)):
            t_tensor = torch.full((batch_size, 1), float(t_target), device=device, dtype=dtype)
        elif isinstance(t_target, torch.Tensor):
            t_tensor = t_target.to(device=device, dtype=dtype).view(batch_size, 1)
        else: raise TypeError(f"不支持的目标时间类型: {type(t_target)}")
        return t_tensor * 0.01 # 简单缩放

    def _fuse_time_features(self, spatial_features, time_features):
        """融合时间特征到空间特征 (简化调制)"""
        batch_size = spatial_features.shape[0]
        time_channel = time_features.view(batch_size, 1, 1, 1).to(spatial_features.dtype)
        one_tensor = torch.tensor(1.0, device=spatial_features.device, dtype=spatial_features.dtype)
        return spatial_features * (one_tensor + time_channel)

    def _process_with_cnn(self, initial_state, k_field, u_field, t_target):
        """使用CNN处理（通常是小尺寸或基础分辨率）"""
        device = initial_state.device
        dtype = initial_state.dtype
        batch_size = initial_state.shape[0]

        # 确保输入形状 [B, 1, H, W]
        if initial_state.ndim == 3: initial_state = initial_state.unsqueeze(1)
        # k_field, u_field should already be [B, 1, H, W] from _ensure_shape

        # 连接输入通道
        cnn_input = torch.cat([initial_state, k_field, u_field], dim=1) # [B, 3, H, W]

        # 时间编码
        t_encoded = self._encode_time(t_target, batch_size, device, dtype)

        # CNN处理
        features = self.encoder(cnn_input)
        fused_features = self._fuse_time_features(features, t_encoded)
        # Return dictionary with decoded state and derivative
        outputs = {}
        if self.output_state:
            outputs['state'] = self.decoder(fused_features)
        if self.output_derivative:
            outputs['derivative'] = self.derivative_decoder(fused_features)
        return outputs

    def _process_multi_resolution(self, initial_state, k_field, u_field, t_target, original_shape):
        """多分辨率处理中等尺寸输入"""
        # 下采样
        initial_state_down = self.downsampler(initial_state)
        k_field_down = self.downsampler(k_field)
        u_field_down = self.downsampler(u_field)

        # 在基础分辨率处理 (returns dict)
        output_dict_down = self._process_with_cnn(initial_state_down, k_field_down, u_field_down, t_target)

        # 上采样字典中的每个张量回原始分辨率
        output_dict_up = {}
        upsampler = nn.Upsample(size=original_shape, mode='bilinear', align_corners=False)
        for key, tensor_down in output_dict_down.items():
            output_dict_up[key] = upsampler(tensor_down)

        return output_dict_up

    def _process_tiled(self, initial_state, k_field, u_field, t_target, original_shape, tile_size=None, overlap=0.1):
        """分块处理超大尺寸输入 (带重叠)"""
        if tile_size is None: tile_size = self.base_resolution
        height, width = original_shape
        batch_size = initial_state.shape[0]
        device = initial_state.device
        dtype = initial_state.dtype

        # 计算重叠像素
        overlap_pixels = int(tile_size * overlap)
        stride = tile_size - overlap_pixels

        # 创建结果字典和计数器（用于平均重叠区域）
        result_dict = {} # Store results per output key ('state', 'derivative')
        # Determine output keys based on model settings
        output_keys = []
        if self.output_state: output_keys.append('state')
        if self.output_derivative: output_keys.append('derivative')
        if not output_keys: raise ValueError("Tiled processing requires at least one output ('state' or 'derivative')")

        for key in output_keys:
            result_dict[key] = torch.zeros((batch_size, self.output_dim, height, width), device=device, dtype=dtype)
        counts = torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)
        
        # 使用 Hann 窗口进行平滑拼接，添加边缘缓和处理
        window = torch.hann_window(tile_size, periodic=False, device=device, dtype=dtype)
        # 调整窗口函数，增强中心区域权重，减少边缘效应
        window = window**0.75  # 幂次调整使窗口中心更平坦，边缘更陡峭
        window2d = window[:, None] * window[None, :] # H, W

        # 修改循环逻辑，确保包含整个图像，避免漏掉末尾行列
        # 计算需要处理的起始位置以覆盖整个图像
        h_starts = list(range(0, height, stride))
        w_starts = list(range(0, width, stride))
        # 确保最后一个块能覆盖到图像边缘
        if h_starts[-1] + tile_size < height:
            h_starts.append(height - tile_size)
        if w_starts[-1] + tile_size < width:
            w_starts.append(width - tile_size)

        for h_start in h_starts:
            for w_start in w_starts:
                # 确保不超出边界
                h_start = min(h_start, height - 1)
                w_start = min(w_start, width - 1)
                
                h_end = min(h_start + tile_size, height)
                w_end = min(w_start + tile_size, width)
                
                # 计算当前块的实际尺寸
                current_tile_h = h_end - h_start
                current_tile_w = w_end - w_start

                # 提取块
                h_slice = slice(h_start, h_end)
                w_slice = slice(w_start, w_end)

                initial_tile = initial_state[:, :, h_slice, w_slice]
                k_tile = k_field[:, :, h_slice, w_slice]
                u_tile = u_field[:, :, h_slice, w_slice]

                # 如果块尺寸小于 tile_size，需要填充
                if current_tile_h < tile_size or current_tile_w < tile_size:
                     # 使用反射填充，更好地保持边界连续性
                     pad_h = tile_size - current_tile_h
                     pad_w = tile_size - current_tile_w
                     initial_tile = F.pad(initial_tile, (0, pad_w, 0, pad_h), mode='reflect')
                     k_tile = F.pad(k_tile, (0, pad_w, 0, pad_h), mode='reflect')
                     u_tile = F.pad(u_tile, (0, pad_w, 0, pad_h), mode='reflect')

                # 处理块 (returns dict)
                tile_output_dict = self._process_with_cnn(initial_tile, k_tile, u_tile, t_target)

                # 如果填充了，裁剪回原始块尺寸
                if current_tile_h < tile_size or current_tile_w < tile_size:
                     for key in tile_output_dict:
                          if key in tile_output_dict:
                              tile_output_dict[key] = tile_output_dict[key][:, :, :current_tile_h, :current_tile_w]
                          else:
                              logging.warning(f"Key '{key}' not found in tile output during cropping.")

                # 获取当前块的窗口，调整大小匹配当前块
                current_window = window2d[:current_tile_h, :current_tile_w].view(1, 1, current_tile_h, current_tile_w)

                # 加权累加结果
                for key in output_keys:
                    if key in tile_output_dict:
                         # 确保形状匹配
                         target_slice = result_dict[key][:, :, h_slice, w_slice]
                         tile_res = tile_output_dict[key]
                         win = current_window
                         
                         if target_slice.shape[-2:] != tile_res.shape[-2:] or target_slice.shape[-2:] != win.shape[-2:]:
                              raise RuntimeError(f"Shape mismatch during tiled accumulation for key '{key}'. "
                                                 f"Target: {target_slice.shape}, Tile: {tile_res.shape}, Window: {win.shape}")
                         # 累加带权重的结果
                         result_dict[key][:, :, h_slice, w_slice] += tile_res * win
                    else:
                         logging.warning(f"Key '{key}' expected but not found in tile output dictionary during tiling.")
                
                # 累加窗口权重，用于后续归一化
                counts[:, :, h_slice, w_slice] += current_window

        # 添加额外的检查以防零除
        zero_counts = counts < 1e-8
        if zero_counts.any():
            logging.warning(f"Found {zero_counts.sum().item()} pixels with zero or near-zero weight counts. Adding bias.")
            counts = torch.where(zero_counts, torch.ones_like(counts) * 1e-8, counts)
        
        # 平均重叠区域
        final_output_dict = {}
        for key in output_keys:
            final_output_dict[key] = result_dict[key] / counts  # 平均权重

        # 返回结果
        if len(final_output_dict) == 1:
            return next(iter(final_output_dict.values()))
        return final_output_dict

    def _predict_state_adaptive(self, initial_state, params, t_target):
        """优化的网格状态预测，支持多分辨率处理"""
        input_shape = initial_state.shape[-2:] # H, W
        batch_size = initial_state.shape[0]
        device = initial_state.device
        dtype = initial_state.dtype

        # 提取并确保参数形状
        k_field = self._ensure_shape(params.get('K'), input_shape, batch_size, device, dtype)
        u_field = self._ensure_shape(params.get('U'), input_shape, batch_size, device, dtype)

        # 根据尺寸选择处理策略
        if max(input_shape) <= self.base_resolution:
            return self._process_with_cnn(initial_state, k_field, u_field, t_target)
        elif max(input_shape) <= self.max_resolution:
            return self._process_multi_resolution(initial_state, k_field, u_field, t_target, input_shape)
        else:
            logging.info(f"Input size {input_shape} > max_resolution {self.max_resolution}. Using tiled processing.")
            # tile_size and overlap could be configurable
            return self._process_tiled(initial_state, k_field, u_field, t_target, input_shape, tile_size=self.base_resolution, overlap=0.1)


    def forward(self, x, mode='predict_state'):
        """
        前向传播，支持双输出和不同模式。

        Args:
            x: 输入数据 (字典或元组，取决于模式)
            mode (str): 'predict_coords' 或 'predict_state'

        Returns:
            dict or torch.Tensor: 包含 'state' 和/或 'derivative' 的字典，
                                  或单个张量（如果只请求一个输出）。
        """
        outputs = {}

        if mode == 'predict_coords':
            # 1. 准备并标准化坐标和参数输入
            if not isinstance(x, dict): raise TypeError("对于 'predict_coords' 模式，输入 x 必须是字典。")
            
            # 使用统一的坐标标准化函数
            from .utils import standardize_coordinate_system
            coords = standardize_coordinate_system(
                x, 
                domain_x=self.domain_x, 
                domain_y=self.domain_y,
                normalize=True  # 归一化到 [0,1] 范围
            )
            
            # 使用归一化坐标采样参数
            x_coords_norm = coords['x']
            y_coords_norm = coords['y']
            k_value = self._sample_at_coords(x.get('k_grid'), x_coords_norm, y_coords_norm)
            u_value = self._sample_at_coords(x.get('u_grid'), x_coords_norm, y_coords_norm)
            augmented_coords = {**coords, 'k': k_value, 'u': u_value}

            # 2. 通过基础 MLP 获取共享特征
            # Need to call the base MLP's forward method correctly
            mlp_features = self.coordinate_mlp_base(augmented_coords, mode='predict_coords') # Pass dict

            # 3. 通过各自的头计算输出
            if self.output_state:
                outputs['state'] = self.state_head(mlp_features)
            if self.output_derivative:
                outputs['derivative'] = self.derivative_head(mlp_features)

        elif mode == 'predict_state':
            # 1. 解析输入
            if isinstance(x, dict):
                initial_state = x.get('initial_state')
                params = x.get('params')
                t_target = x.get('t_target')
            elif isinstance(x, (tuple, list)) and len(x) == 3:
                initial_state, params, t_target = x
            else: raise ValueError("对于 'predict_state' 模式，输入 x 必须是字典或 (initial_state, params, t_target) 元组/列表")
            if initial_state is None or params is None or t_target is None: raise ValueError("缺少 'initial_state', 'params', 或 't_target' 输入")

            # 2. 调用自适应状态预测逻辑获取共享特征
            # _predict_state_adaptive now returns the final output dictionary
            outputs = self._predict_state_adaptive(initial_state, params, t_target)
            # No need to apply decoders again here
        else:
            raise ValueError(f"未知的 forward 模式: {mode}")

        # 根据请求的输出数量返回
        if len(outputs) == 0:
             raise ValueError("模型未配置为输出任何内容 (state=False, derivative=False)")
        elif len(outputs) == 1:
             return next(iter(outputs.values())) # Return the single tensor
        else:
             return outputs # Return the dictionary

# ADDED: Create a TimeDerivative version of the basic MLP_PINN model
class TimeDerivativeMLP_PINN(TimeDerivativePINN):
    """支持同时输出状态及其时间导数的简单MLP版本"""
    
    def __init__(self, input_dim=3, output_dim=1, hidden_layers=8, hidden_neurons=256, activation=nn.Tanh()):
        super().__init__()
        self.output_dim = output_dim
        
        # 共享特征提取网络
        self.feature_layers = nn.Sequential()
        
        # 输入层
        self.feature_layers.append(nn.Linear(input_dim, hidden_neurons))
        self.feature_layers.append(activation)
        
        # 隐藏层
        for _ in range(hidden_layers - 1):
            self.feature_layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            self.feature_layers.append(activation)
        
        # 状态和导数的输出头
        self.state_head = nn.Linear(hidden_neurons, output_dim)
        self.derivative_head = nn.Linear(hidden_neurons, output_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.feature_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 初始化输出头
        nn.init.xavier_uniform_(self.state_head.weight)
        nn.init.constant_(self.state_head.bias, 0)
        
        nn.init.xavier_uniform_(self.derivative_head.weight)
        nn.init.constant_(self.derivative_head.bias, 0)
    
    def _prepare_input(self, x, mode):
        """准备模型输入"""
        if mode == 'predict_coords':
            if not isinstance(x, dict):
                raise TypeError("对于'predict_coords'模式，输入x必须是字典")
            
            # 提取必要的坐标
            expected_keys = ['x', 'y', 't']
            tensors_to_cat = []
            
            for key in expected_keys:
                if key not in x:
                    raise ValueError(f"缺少必需的坐标键'{key}'")
                tensors_to_cat.append(x[key])
            
            # 如果需要额外的输入维度，例如参数
            if hasattr(self, 'input_dim') and self.input_dim > 3:
                for i in range(3, self.input_dim):
                    key = f'param{i-2}'
                    if key in x:
                        tensors_to_cat.append(x[key])
                    else:
                        # 使用零张量作为默认值
                        tensors_to_cat.append(torch.zeros_like(x['x']))
            
            return torch.cat(tensors_to_cat, dim=-1)
            
        elif mode == 'predict_state':
            # 实现与MLP_PINN类似的网格状态预测逻辑
            if not isinstance(x, dict):
                raise TypeError("对于'predict_state'模式，输入x必须是字典")
            
            initial_state = x.get('initial_state')
            params = x.get('params', {})
            t_target = x.get('t_target')
            
            if initial_state is None or t_target is None:
                raise ValueError("对于'predict_state'模式，缺少'initial_state'或't_target'输入")
            
            batch_size, _, height, width = initial_state.shape
            device = initial_state.device
            dtype = initial_state.dtype
            
            # 创建网格坐标
            y_coords = torch.linspace(0, 1, height, device=device, dtype=dtype)
            x_coords = torch.linspace(0, 1, width, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # 展平为点列表
            x_flat = grid_x.reshape(-1, 1).expand(batch_size, -1, 1)
            y_flat = grid_y.reshape(-1, 1).expand(batch_size, -1, 1)
            
            # 准备时间输入
            if isinstance(t_target, (int, float)):
                t_flat = torch.full((batch_size, height*width, 1), float(t_target), device=device, dtype=dtype)
            elif isinstance(t_target, torch.Tensor):
                t_flat = t_target.view(batch_size, 1, 1).expand(-1, height*width, 1)
            else:
                raise TypeError(f"不支持的t_target类型: {type(t_target)}")
            
            # 组合输入
            model_input = torch.cat([x_flat, y_flat, t_flat], dim=-1)
            
            # 如果使用了额外参数
            if hasattr(self, 'input_dim') and self.input_dim > 3:
                for i in range(3, self.input_dim):
                    param_name = f'param{i-2}'
                    param_val = params.get(param_name, 0.0)
                    
                    if isinstance(param_val, (int, float)):
                        param_tensor = torch.full((batch_size, height*width, 1), float(param_val), device=device, dtype=dtype)
                    elif isinstance(param_val, torch.Tensor):
                        # TODO: 处理张量参数
                        param_tensor = param_val.expand(batch_size, height*width, 1)
                    else:
                        raise TypeError(f"不支持的参数类型'{param_name}': {type(param_val)}")
                    
                    model_input = torch.cat([model_input, param_tensor], dim=-1)
            
            return model_input, (batch_size, height, width)
        
        else:
            raise ValueError(f"未知的forward模式: {mode}")
    
    def forward(self, x, mode='predict_coords'):
        """前向传播，支持状态和导数双输出"""
        if mode == 'predict_coords':
            # 预测坐标点处的值
            model_input = self._prepare_input(x, mode)
            features = self.feature_layers(model_input)
            
            # 根据输出模式返回结果
            outputs = {}
            if self.output_state:
                outputs['state'] = self.state_head(features)
            if self.output_derivative:
                outputs['derivative'] = self.derivative_head(features)
            
            # 根据需要返回字典或单一张量
            if len(outputs) == 1:
                return next(iter(outputs.values()))
            else:
                return outputs
                
        elif mode == 'predict_state':
            # 预测整个状态网格
            model_input, shape_info = self._prepare_input(x, mode)
            batch_size, height, width = shape_info
            
            # 处理展平的输入
            features = self.feature_layers(model_input)
            
            # 获取输出
            outputs = {}
            if self.output_state:
                state_flat = self.state_head(features)
                outputs['state'] = state_flat.reshape(batch_size, height, width, self.output_dim).permute(0, 3, 1, 2)
            if self.output_derivative:
                derivative_flat = self.derivative_head(features)
                outputs['derivative'] = derivative_flat.reshape(batch_size, height, width, self.output_dim).permute(0, 3, 1, 2)
            
            # 根据需要返回字典或单一张量
            if len(outputs) == 1:
                return next(iter(outputs.values()))
            else:
                return outputs
        
        else:
            raise ValueError(f"未知的forward模式: {mode}")
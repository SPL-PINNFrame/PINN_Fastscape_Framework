import numpy as np
import torch
import torch.nn.functional as F
import math # For sqrt(2)
import logging
from .utils import ErrorHandler, ensure_tensor, validate_and_transform_shape

# -----------------------------------------------------------------------------
# Terrain Derivatives (Differentiable using PyTorch Conv2d)
# -----------------------------------------------------------------------------

def get_sobel_kernels(dx, dy):
    """Gets Sobel kernels for gradient calculation."""
    # Sobel kernels for d/dx and d/dy
    # Note the scaling by 1/(8*dx) or 1/(8*dy) as in TerrainDerivatives.f90
    # Ensure dx and dy are floats before using them in calculations
    dx_float = float(dx)
    dy_float = float(dy)
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / (8.0 * dx_float)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32) / (8.0 * dy_float)
    # Reshape for Conv2d: (out_channels, in_channels, height, width)
    kernel_x = kernel_x.view(1, 1, 3, 3)
    kernel_y = kernel_y.view(1, 1, 3, 3)
    return kernel_x, kernel_y

def calculate_slope_magnitude(h, dx, dy, padding_mode='replicate'):
    """
    Calculates the magnitude of the terrain slope using Sobel operators.
    Matches the finite difference scheme in fastscapelib-fortran/src/TerrainDerivatives.f90 slope subroutine.

    Args:
        h (torch.Tensor): Topography tensor (batch, 1, height, width).
        dx (float): Grid spacing in x direction.
        dy (float): Grid spacing in y direction.
        padding_mode (str): Padding mode for convolution.

    Returns:
        torch.Tensor: Slope magnitude tensor (batch, 1, height, width).
                      Note: This is the magnitude |∇h|, not atan(|∇h|).
    """
    # 确保输入是tensor，并具有一致的数据类型
    h = ensure_tensor(h)
    device = h.device
    dtype = h.dtype
    
    kernel_x, kernel_y = get_sobel_kernels(dx, dy)
    # Ensure kernel dtype matches input tensor h's dtype
    kernel_x = kernel_x.to(device=device, dtype=dtype)
    kernel_y = kernel_y.to(device=device, dtype=dtype)

    # Calculate gradients using convolution
    # padding='same' requires PyTorch 1.9+ and ensures output size matches input size
    # For older versions, use padding=1 and potentially crop the output if needed.
    # Using F.pad and then conv2d with padding=0 is another option.
    pad_size = 1 # for 3x3 kernel
    h_padded = F.pad(h, (pad_size, pad_size, pad_size, pad_size), mode=padding_mode)

    dzdx = F.conv2d(h_padded, kernel_x, padding=0)
    dzdy = F.conv2d(h_padded, kernel_y, padding=0)

    # Calculate slope magnitude: sqrt((dz/dx)^2 + (dz/dy)^2)
    slope_mag = torch.sqrt(dzdx**2 + dzdy**2 + 1e-10) # Add epsilon for numerical stability

    return slope_mag

def get_laplacian_kernel(dx, dy):
    """Gets the 5-point finite difference kernel for the Laplacian."""
    # Assuming dx == dy for the standard 5-point stencil scaling
    # If dx != dy, the formula is more complex:
    # (h(i+1,j)-2h(i,j)+h(i-1,j))/dx^2 + (h(i,j+1)-2h(i,j)+h(i,j-1))/dy^2
    # For simplicity, assume dx=dy here. If not, this needs adjustment.
    # assert abs(dx - dy) < 1e-6, "Laplacian kernel currently assumes dx == dy"
    # Kernel for d^2/dx^2 + d^2/dy^2
    # [[ 0,  1,  0],
    #  [ 1, -4,  1],
    #  [ 0,  1,  0]] / dx^2 (or dy^2)
    scale = 1.0 / (dx * dx) # Assuming dx == dy
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32) * scale
    # Reshape for Conv2d: (out_channels, in_channels, height, width)
    kernel = kernel.view(1, 1, 3, 3)
    return kernel

def calculate_laplacian(h, dx, dy, padding_mode='replicate'):
    """
    Calculates the Laplacian of the topography using a 5-point finite difference stencil.
    This is the term used in the linear diffusion equation: dh/dt = Kd * Laplacian(h).

    Args:
        h (torch.Tensor): Topography tensor (batch, 1, height, width).
        dx (float): Grid spacing in x direction.
        dy (float): Grid spacing in y direction.
        padding_mode (str): Padding mode for convolution.

    Returns:
        torch.Tensor: Laplacian tensor (batch, 1, height, width).
    """
    # 确保输入是tensor并具有一致的数据类型
    h = ensure_tensor(h)
    device = h.device
    dtype = h.dtype
    
    # TODO: Handle dx != dy case properly if needed.
    if abs(dx - dy) > 1e-6:
        print("Warning: calculate_laplacian currently assumes dx == dy for simplicity.")
        # Implement the more general formula if dx != dy
        # lap = (h(i+1,j)-2h(i,j)+h(i-1,j))/dx^2 + (h(i,j+1)-2h(i,j)+h(i,j-1))/dy^2
        # This can be done with two separate convolutions for d^2/dx^2 and d^2/dy^2
        kernel_dxx = torch.tensor([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=h.dtype).view(1, 1, 3, 3) / (dx**2)
        kernel_dyy = torch.tensor([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=h.dtype).view(1, 1, 3, 3) / (dy**2)
        kernel_dxx = kernel_dxx.to(device)
        kernel_dyy = kernel_dyy.to(device)
        pad_size = 1
        h_padded = F.pad(h, (pad_size, pad_size, pad_size, pad_size), mode=padding_mode)
        lap_x = F.conv2d(h_padded, kernel_dxx, padding=0)
        lap_y = F.conv2d(h_padded, kernel_dyy, padding=0)
        laplacian = lap_x + lap_y
    else:
        # Use the simpler 5-point kernel assuming dx == dy
        kernel = get_laplacian_kernel(dx, dy)
        kernel = kernel.to(device=device, dtype=dtype)
        pad_size = 1
        h_padded = F.pad(h, (pad_size, pad_size, pad_size, pad_size), mode=padding_mode)
        laplacian = F.conv2d(h_padded, kernel, padding=0)

    return laplacian

# -----------------------------------------------------------------------------
# Flow Routing and Drainage Area (Differentiable Approximation - CHALLENGE)
# -----------------------------------------------------------------------------

def calculate_drainage_area_differentiable_optimized(h, dx, dy, precip=1.0, temp=0.01, num_iters=10, verbose=False, 
                                                     dynamic_temp=True, depression_handling='weighted_aggregation'): 
    """优化的可微分汇水面积计算，使用PyTorch的内置操作减少循环
    
    Args:
        h (torch.Tensor): 地形高度 (batch, 1, height, width)
        dx (float): x方向网格间距
        dy (float): y方向网格间距
        precip (float or torch.Tensor): 降水输入（标量或与h形状匹配的张量）
        temp (float): softmax温度参数，影响流向分配的"锐度"
        num_iters (int): 流量累积迭代次数
        verbose (bool): 是否打印详细信息
        dynamic_temp (bool): 是否根据坡度分布动态调整温度
        depression_handling (str): 洼地处理策略 ('weighted_aggregation', 'simple')
        
    Returns:
        torch.Tensor: 汇水面积张量 (batch, 1, height, width)
    """
    # 创建错误处理器
    error_handler = ErrorHandler(max_retries=1)
    
    # 确保输入是tensor，并验证形状
    h = ensure_tensor(h)
    batch_size, _, height, width = h.shape
    device = h.device
    dtype = h.dtype
    cell_area = dx * dy

    # Handle precipitation input (scalar or tensor)
    if isinstance(precip, (float, int)):
        local_flow = torch.full_like(h, precip * cell_area)
    elif isinstance(precip, torch.Tensor):
        precip = ensure_tensor(precip, device=device, dtype=dtype)
        if precip.shape == h.shape:
             local_flow = precip * cell_area
        elif precip.numel() == 1: # Scalar tensor
             local_flow = torch.full_like(h, precip.item() * cell_area)
        else:
             # Allow broadcasting if precip shape is (b, 1, 1, 1) or similar
             try:
                 local_flow = precip * cell_area
             except RuntimeError:
                  raise ValueError(f"Shape mismatch between precip {precip.shape} and h {h.shape}, cannot broadcast.")
    else:
        raise TypeError("precip must be float or torch.Tensor")

    # 1. 计算高度差和坡度
    # Use padding from torch.nn
    pad = torch.nn.ReplicationPad2d(1)
    h_pad = pad(h)

    # 创建8个方向的卷积核 (h_center - h_neighbor)
    kernels = torch.zeros(8, 1, 3, 3, device=device, dtype=dtype)
    # N, NE, E, SE, S, SW, W, NW
    # Note: Kernel indices (y, x) relative to top-left (0,0) of 3x3 kernel
    directions = [(0,1), (0,2), (1,2), (2,2), (2,1), (2,0), (1,0), (0,0)]
    for i, (y, x) in enumerate(directions):
        kernels[i, 0, 1, 1] = 1  # 中心点
        kernels[i, 0, y, x] = -1 # 邻居点

    # 计算高度差
    dh = F.conv2d(h_pad, kernels, padding=0)  # [B, 8, H, W]

    # 计算距离
    dist_diag = math.sqrt(dx**2 + dy**2)
    distances = torch.tensor([dy, dist_diag, dx, dist_diag,
                             dy, dist_diag, dx, dist_diag],
                             device=device, dtype=dtype).view(1, 8, 1, 1)

    # 计算坡度
    slopes = dh / (distances + 1e-10)

    # 2. 计算流向权重 (Softmax)
    # 如果启用了动态温度，根据坡度分布调整
    if dynamic_temp:
        # 分析坡度统计，自适应调整温度
        # 使用负坡度的百分位数来调整温度
        valid_slopes = slopes[slopes < 0]
        if valid_slopes.numel() > 0:
            # 计算10%和90%分位数，获取坡度范围
            q10 = torch.quantile(valid_slopes, 0.1)
            q90 = torch.quantile(valid_slopes, 0.9)
            slope_range = q90 - q10
            # 调整温度：坡度范围小（平坦地形）时使用更高温度
            # 坡度范围大（陡峭地形）时使用更低温度
            adjusted_temp = temp * (0.1 / (slope_range.abs() + 0.01))
            # 限制温度在合理范围内
            adjusted_temp = torch.clamp(adjusted_temp, min=temp/10, max=temp*10)
            if verbose:
                logging.info(f"Dynamic temperature adjustment: original={temp}, adjusted={adjusted_temp.item()}")
            temp = adjusted_temp.item()
    
    # 带错误处理的softmax计算函数
    @error_handler.catch_and_handle(retry_on=[RuntimeError], ignore=[])
    def compute_flow_weights(slope_values, temperature):
        # 仅考虑下坡流向 (negative slopes)
        # Use -slopes in softmax, clamp positive slopes to ensure near-zero probability
        softmax_input = -slope_values / temperature
        softmax_input = torch.where(slopes >= 0, 
                                   torch.full_like(softmax_input, -torch.finfo(softmax_input.dtype).max), 
                                   softmax_input)
        # Clamp to prevent overflow in exp()
        softmax_input = torch.clamp(softmax_input, max=80)
        return F.softmax(softmax_input, dim=1)  # [B, 8, H, W]
    
    # 首先尝试正常的温度
    weights = compute_flow_weights(slopes, temp)
    
    # 如果计算失败或包含NaN，尝试使用更高的温度
    if weights is None or torch.isnan(weights).any():
        logging.warning(f"在温度 {temp} 下计算流向权重失败或产生了NaN值。尝试更高的温度...")
        higher_temp = temp * 10  # 增加温度使softmax更平滑
        weights = compute_flow_weights(slopes, higher_temp)
        
        if weights is None or torch.isnan(weights).any():
            logging.error(f"即使在更高的温度 {higher_temp} 下，计算流向权重仍然失败。使用均匀权重。")
            # 创建均匀权重作为后备
            weights = torch.full((batch_size, 8, height, width), 1.0/8.0, device=device, dtype=dtype)

    # 处理仍可能存在的NaN
    if torch.isnan(weights).any():
        nan_mask = torch.isnan(weights)
        nan_count = nan_mask.sum().item()
        logging.warning(f"权重中检测到 {nan_count} 个NaN值。替换为平均值(1/8)。")
        weights = torch.where(nan_mask, torch.full_like(weights, 1.0/8.0), weights)
        # 重新归一化
        weights_sum = weights.sum(dim=1, keepdim=True)
        weights = weights / (weights_sum + 1e-12)

    # 3. 洼地处理：识别局部最小值（洼地）
    h_pad = F.pad(h, (1, 1, 1, 1), mode='constant', value=float('inf'))
    is_min = torch.ones_like(h, dtype=torch.bool)
    
    # 检查8个相邻点，确定每个点是否为局部最小值
    offsets = [
        (1, 0), (1, -1), (0, -1), (-1, -1),
        (-1, 0), (-1, 1), (0, 1), (1, 1)
    ]
    for dy_offset, dx_offset in offsets:
        neighbor_h = h_pad[:, :, 
                        1+dy_offset:height+1+dy_offset, 
                        1+dx_offset:width+1+dx_offset]
        is_min = is_min & (h <= neighbor_h)
    
    # 创建洼地掩码
    depression_mask = is_min.float()
    
    # 4. 迭代计算汇水面积，根据depression_handling策略处理洼地
    drainage_area = local_flow.clone()
    
    # 预计算反向偏移和索引 (for optimized accumulation)
    # Offsets (dy, dx) from neighbor to center
    offsets = [
        (1, 0), (1, -1), (0, -1), (-1, -1),
        (-1, 0), (-1, 1), (0, 1), (1, 1)
    ]
    # Weight index at neighbor that points to center
    # N neighbor (offset (1,0)) uses weight[4] (S)
    # NE neighbor (offset (1,-1)) uses weight[5] (SW) ... etc.
    reverse_weight_indices = [4, 5, 6, 7, 0, 1, 2, 3]

    # 使用错误处理上下文进行迭代计算
    with error_handler.handling_context(
        retry_on=[RuntimeError], 
        ignore=[], 
        reraise=False,
        max_retries=2
    ) as err_ctx:
        
        # 优化：增加迭代次数以提高准确性
        actual_iters = min(max(num_iters, 30), 100)  # 确保迭代次数足够但不过多
        
        if verbose: 
            logging.info(f"使用简化迭代流量累积 ({actual_iters} 迭代)。")
            
        try:
            for _ in range(actual_iters):
                # 检查是否需要重试
                if err_ctx.retries > 0:
                    # 如果是重试，简化计算或使用不同的策略
                    logging.info("使用稳定化迭代计算汇水面积...")
                    
                inflow = torch.zeros_like(drainage_area)
                # Pad current drainage area to access neighbor values easily
                da_padded = F.pad(drainage_area, (1, 1, 1, 1), mode='constant', value=0)

                for i in range(8):
                    # Get the weight assigned by neighbor 'i' towards the center
                    neighbor_weight_to_center = weights[:, reverse_weight_indices[i]:reverse_weight_indices[i]+1, :, :]

                    # Get the drainage area of neighbor 'i' by shifting the padded DA
                    dy_offset, dx_offset = offsets[i]
                    neighbor_da = da_padded[:, :, 1+dy_offset:height+1+dy_offset, 1+dx_offset:width+1+dx_offset]

                    # Calculate inflow from neighbor 'i'
                    inflow += neighbor_da * neighbor_weight_to_center

                # 洼地处理策略
                if depression_handling == 'weighted_aggregation':
                    # 加权汇聚：洼地从周围收集流量，但以较小的权重输出
                    # 这样洼地内的水会累积但不会完全困住
                    depression_factor = torch.where(
                        depression_mask > 0, 
                        torch.full_like(depression_mask, 0.5),  # 洼地的传递因子
                        torch.ones_like(depression_mask)  # 非洼地的传递因子
                    )
                    
                    # 洼地接收额外流量
                    depression_bonus = torch.where(
                        depression_mask > 0,
                        inflow * 0.3,  # 额外收集的流量
                        torch.zeros_like(inflow)
                    )
                    
                    # 更新汇水面积：本地流量 + 调整后的流入 + 洼地额外流量
                    drainage_area = local_flow + inflow * depression_factor + depression_bonus
                
                elif depression_handling == 'simple':
                    # 简单处理：所有单元都正常处理
                    drainage_area = local_flow + inflow
                
                else:
                    raise ValueError(f"未知的洼地处理策略: {depression_handling}")
                
                # 稳定性检查
                if torch.isnan(drainage_area).any():
                    raise RuntimeError("汇水面积计算中检测到NaN值，正在重试...")
                    
                if torch.isinf(drainage_area).any():
                    raise RuntimeError("汇水面积计算中检测到Inf值，正在重试...")
        
        except Exception as e:
            logging.error(f"汇水面积计算失败: {str(e)}")
            # 发生错误时，返回一个简单的估计
            # 对于简单表面，这可能是合理的近似
            drainage_area = local_flow.clone()
            
            # 为洼地创建一个简单的汇聚模式
            # 查找局部最小值
            h_pad = F.pad(h, (1, 1, 1, 1), mode='constant', value=float('inf'))
            is_min = torch.ones_like(h, dtype=torch.bool)
            
            # 检查8个相邻点，确定每个点是否为局部最小值
            for dy_offset, dx_offset in offsets:
                neighbor_h = h_pad[:, :, 
                                1+dy_offset:height+1+dy_offset, 
                                1+dx_offset:width+1+dx_offset]
                is_min = is_min & (h <= neighbor_h)
            
            # 对局部最小值进行特殊处理（增加汇聚）
            min_mask = is_min.float() * 5.0  # 增强因子
            drainage_area = drainage_area * (1.0 + min_mask)
            
            logging.warning("使用简化的汇水面积估计。")

    return drainage_area


# -----------------------------------------------------------------------------
# Physics Components (Stream Power, Diffusion)
# -----------------------------------------------------------------------------

def stream_power_erosion(h, drainage_area, slope_magnitude, K_f, m, n):
    """
    Calculates the erosion rate based on the Stream Power Law.
    E = K_f * A^m * S^n

    Args:
        h (torch.Tensor): Topography (used for potential masking, e.g., below sea level).
        drainage_area (torch.Tensor): Drainage area A.
        slope_magnitude (torch.Tensor): Slope magnitude S.
        K_f (float or torch.Tensor): Erodibility coefficient.
        m (float): Drainage area exponent.
        n (float): Slope exponent.

    Returns:
        torch.Tensor: Erosion rate tensor (positive values indicate erosion).
    """
    # 确保输入是tensor，并具有一致的数据类型
    h = ensure_tensor(h)
    drainage_area = ensure_tensor(drainage_area, device=h.device, dtype=h.dtype)
    slope_magnitude = ensure_tensor(slope_magnitude, device=h.device, dtype=h.dtype)
    
    if isinstance(K_f, (int, float)):
        K_f = torch.tensor(K_f, device=h.device, dtype=h.dtype)
    elif isinstance(K_f, torch.Tensor):
        K_f = K_f.to(device=h.device, dtype=h.dtype)
    
    # Add small epsilon to avoid issues with A=0 or S=0 if m or n are non-integer or negative.
    epsilon = 1e-10
    erosion_rate = K_f * (drainage_area + epsilon)**m * (slope_magnitude + epsilon)**n
    # Optional: Mask erosion below sea level if needed
    # erosion_rate[h < sea_level] = 0
    return erosion_rate

def hillslope_diffusion(h, K_d, dx, dy, padding_mode='replicate'):
    """
    Calculates the change in elevation due to linear hillslope diffusion.
    D = Kd * Laplacian(h)

    Args:
        h (torch.Tensor): Topography tensor.
        K_d (float or torch.Tensor): Diffusivity coefficient.
        dx (float): Grid spacing in x.
        dy (float): Grid spacing in y.
        padding_mode (str): Padding mode for Laplacian calculation.

    Returns:
        torch.Tensor: Diffusion rate tensor.
    """
    # 确保输入是tensor，并具有一致的数据类型
    h = ensure_tensor(h)
    
    if isinstance(K_d, (int, float)):
        K_d = torch.tensor(K_d, device=h.device, dtype=h.dtype)
    elif isinstance(K_d, torch.Tensor):
        K_d = K_d.to(device=h.device, dtype=h.dtype)
    
    laplacian_h = calculate_laplacian(h, dx, dy, padding_mode=padding_mode)
    diffusion_rate = K_d * laplacian_h
    return diffusion_rate

# -----------------------------------------------------------------------------
# Combined PDE Right Hand Side
# -----------------------------------------------------------------------------

def calculate_dhdt_physics(h, U, K_f, m, n, K_d, dx, dy, precip=1.0, padding_mode='replicate', da_optimize_params=None):
    """
    Calculates the physics-based time derivative of elevation (RHS of the PDE).
    dh/dt = U - E + D
          = U - K_f * A^m * S^n + K_d * Laplacian(h)

    Args:
        h (torch.Tensor): Current topography.
        U (torch.Tensor or float): Uplift rate.
        K_f (float): Stream power erodibility.
        m (float): Stream power area exponent.
        n (float): Stream power slope exponent.
        K_d (float): Hillslope diffusivity.
        dx (float): Grid spacing x.
        dy (float): Grid spacing y.
        precip (float or torch.Tensor): Precipitation for drainage area calculation.
        padding_mode (str): Padding mode for derivatives.
        da_optimize_params (dict, optional): Parameters for the optimized drainage area function
                                             (e.g., {'temp': 0.01, 'num_iters': 50}).

    Returns:
        torch.Tensor: The calculated dh/dt based on physics.
    """
    # 确保输入是tensor，并具有一致的数据类型
    h = ensure_tensor(h)
    device = h.device
    dtype = h.dtype
    
    # 确保U是tensor并匹配h的形状
    if isinstance(U, (int, float)):
        U = torch.tensor(U, device=device, dtype=dtype)
    elif isinstance(U, torch.Tensor):
        U = U.to(device=device, dtype=dtype)
    
    # Calculate slope magnitude using the function defined in this file
    slope_mag = calculate_slope_magnitude(h, dx, dy, padding_mode=padding_mode)

    # Use the optimized drainage area function
    da_params = da_optimize_params if da_optimize_params is not None else {}
    drainage_area = calculate_drainage_area_differentiable_optimized(h, dx, dy, precip=precip, **da_params)

    erosion_rate = stream_power_erosion(h, drainage_area, slope_mag, K_f, m, n)

    # Calculate diffusion using the laplacian function defined in this file
    laplacian_h = calculate_laplacian(h, dx, dy, padding_mode=padding_mode)
    diffusion_rate = K_d * laplacian_h

    # Combine terms
    dhdt = U - erosion_rate + diffusion_rate

    return dhdt


# -----------------------------------------------------------------------------
# Validation Utilities
# -----------------------------------------------------------------------------

def validate_drainage_area(h, dx, dy, pinn_method_params=None, d8_method='fastscape'):
    """比较可微分汇水面积与传统D8算法的精度

    Args:
        h (torch.Tensor): Topography tensor (B, 1, H, W).
        dx (float): Grid spacing x.
        dy (float): Grid spacing y.
        pinn_method_params (dict, optional): Parameters for the differentiable method
                                             (e.g., {'temp': 0.01, 'num_iters': 50}). Defaults to None.
        d8_method (str): Which D8 implementation to use ('fastscape' or potentially others).

    Returns:
        dict: Dictionary containing comparison metrics.
    """
    if pinn_method_params is None:
        pinn_method_params = {'temp': 0.01, 'num_iters': 50} # Default params

    # 获取地形数据的numpy版本(用于传统算法)
    # Assuming batch size B=1 for validation comparison
    if h.shape[0] != 1:
        print("Warning: validate_drainage_area currently assumes batch size 1.")
    h_np = h.squeeze().detach().cpu().numpy() # Remove B, C dims

    # 计算可微分方法的汇水面积
    da_diff_torch = calculate_drainage_area_differentiable_optimized(
        h, dx, dy, **pinn_method_params
    )
    da_diff = da_diff_torch.squeeze().detach().cpu().numpy()

    # --- 使用 xsimlab 计算 D8 汇水面积作为基准 ---
    da_d8 = np.zeros_like(h_np) # Initialize placeholder
    try:
        import xsimlab as xs
        # Import necessary processes from fastscape for flow routing/accumulation
        # The exact import path might vary depending on fastscape version
        try:
            # Try common locations for flow routing components
            from fastscape.processes import (
                FlowAccumulator, # Accumulates flow (drainage area)
                FlowRouter,      # Determines flow directions (D8)
                UniformRectilinearGrid2D, # Correct name for grid process
                SurfaceToErode,  # Process that provides elevation to FlowRouter
                SurfaceTopography, # Process that provides initial topo_elevation
                FastscapelibContext # Provides the Fortran context if needed
            )
        except ImportError:
             # Fallback or alternative paths if the above fails
             # This might need adjustment based on the actual fastscape structure
             print("Warning: Could not import FlowAccumulator/FlowRouter from fastscape.processes. Trying alternatives...")
             # Example: maybe they are directly under fastscape?
             # from fastscape import FlowAccumulator, FlowRouter, UniformGrid2D
             raise # Re-raise the import error if alternatives also fail

        # Define a minimal xsimlab model for D8 drainage area
        # Remove the custom InputTopo process
        # @xs.process
        # class InputTopo:
        #     """Provides the input topography."""
        #     elevation = xs.variable(dims=[('y', 'x')], intent='out')
        #
        #     def initialize(self):
        #         self.elevation = h_np # Use the numpy topography passed to the function

        # Create the model instance
        d8_model = xs.Model({
            'grid': UniformRectilinearGrid2D, # Use correct grid process
            'topo': SurfaceTopography, # Add the process providing initial topography
            'surface': SurfaceToErode, # Use the standard process providing elevation
            'flow': FlowRouter, # D8 flow directions
            'drainage': FlowAccumulator, # Calculates drainage area
            'fs_context': FastscapelibContext # Add the context provider
        })

        # Prepare input dataset for the minimal model
        # Need grid shape and spacing
        ny, nx = h_np.shape
        input_ds = xs.create_setup(
            model=d8_model,
            clocks={'time': [0]}, # Dummy clock, run only once
            input_vars={
                'grid__shape': [ny, nx],
                'grid__dx': dx,
                'grid__dy': dy,
                # Provide the input topography to the SurfaceTopography process
                'topo__elevation': h_np,
                # Link SurfaceTopography output to SurfaceToErode input
                'surface__topo_elevation': xs.foreign(SurfaceTopography, 'elevation'),
                # Link SurfaceToErode output to flow router input
                'flow__elevation': xs.foreign(SurfaceToErode, 'elevation'),
                # Link context to flow router input
                'flow__fs_context': xs.foreign(FastscapelibContext, 'context'),
                # Link flow directions to accumulator
                'drainage__flow_direction': xs.foreign(FlowRouter, 'flow_direction')
            },
            output_vars={'drainage__area': 'time'} # Output the calculated area
        )

        # Run the minimal model
        print("Running minimal xsimlab model for D8 drainage area...")
        result_ds = input_ds.xsimlab.run(model=d8_model)
        da_d8 = result_ds['drainage__area'].squeeze().values # Extract numpy array
        print("Calculated D8 drainage area using xsimlab.")

    except ImportError as e:
        print(f"Warning: Failed to import xsimlab or fastscape components needed for D8 calculation: {e}. Returning zeros for D8 area.")
    except Exception as e:
        print(f"Error computing D8 drainage area using xsimlab: {e}. Returning zeros.")

    # 计算差异指标
    # Avoid division by zero or issues with very small D8 areas
    valid_mask = da_d8 > 1e-8
    if valid_mask.sum() == 0:
        print("Warning: D8 drainage area is zero everywhere. Cannot compute relative error.")
        relative_error = np.full_like(da_diff, np.nan)
    else:
        relative_error = np.full_like(da_diff, np.nan)
        relative_error[valid_mask] = np.abs(da_diff[valid_mask] - da_d8[valid_mask]) / da_d8[valid_mask]

    max_relative_error = np.nanmax(relative_error) if np.any(valid_mask) else np.nan
    mean_relative_error = np.nanmean(relative_error) if np.any(valid_mask) else np.nan

    # 河网匹配度(使用阈值定义河网)
    try:
        # Use a percentile relative to the grid size to avoid issues with flat terrains
        # Or use a fixed area threshold if appropriate for the scale
        threshold_percentile = 95 # Example: top 5% of area defines river network
        if np.any(da_d8 > 1e-8):
             threshold = np.percentile(da_d8[da_d8 > 1e-8], threshold_percentile)
        else:
             threshold = 1e-8 # Default threshold if D8 is all zero

        river_d8 = da_d8 > threshold
        river_diff = da_diff > threshold

        intersection = np.sum(river_d8 & river_diff)
        union = np.sum(river_d8 | river_diff)

        jaccard = intersection / union if union > 0 else np.nan
    except Exception as e:
        print(f"Error calculating Jaccard index: {e}")
        jaccard = np.nan

    # Calculate RMSE and Correlation Coefficient on valid areas
    rmse = np.nan
    correlation = np.nan
    if np.any(valid_mask):
        diff_valid = da_diff[valid_mask]
        d8_valid = da_d8[valid_mask]
        rmse = np.sqrt(np.mean((diff_valid - d8_valid)**2))
        if len(diff_valid) > 1: # Correlation requires at least 2 points
             # Flatten in case they are multidimensional slices
             correlation = np.corrcoef(diff_valid.flatten(), d8_valid.flatten())[0, 1]

    return {
        'max_relative_error': max_relative_error,
        'mean_relative_error': mean_relative_error,
        'rmse': rmse,
        'correlation_coefficient': correlation,
        'river_network_jaccard': jaccard,
        # Optionally return the arrays for visualization
        # 'da_diff': da_diff,
        # 'da_d8': da_d8
    }
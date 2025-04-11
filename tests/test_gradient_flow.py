import sys
import os
import pytest
import torch
import numpy as np

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.losses import (
    compute_grid_temporal_derivative,
    compute_pde_residual_grid_focused,
    compute_pde_residual_dual_output,
    compute_total_loss
)
from src.physics import calculate_dhdt_physics


class SimpleTimeModel(torch.nn.Module):
    """简单的时间相关模型，用于测试梯度流"""
    
    def __init__(self, mode='direct'):
        super().__init__()
        self.mode = mode
        # 创建一个简单的参数，使模型可训练
        self.weight = torch.nn.Parameter(torch.ones(1))
    
    def forward(self, inputs, coords=None):
        """
        参数:
            inputs: 包含't'时间的字典
            coords: 可选的坐标字典
        
        返回:
            根据mode返回不同格式的输出
        """
        if self.mode == 'direct':
            # 直接使时间影响输出（无问题的情况）
            t = inputs['t']
            # 创建一个简单的时间依赖输出: h = w * t^2
            h = self.weight * t**2
            return {'pred': h}
            
        elif self.mode == 'indirect':
            # 间接使时间影响输出（可能会有问题）
            t = inputs['t']
            # 这种情况下autograd可能难以追踪时间依赖: h = w * (t.detach() + 1)^2
            h = self.weight * (t.detach() + 1)**2
            return {'pred': h}
            
        elif self.mode == 'dual_output':
            # 直接预测状态和导数（推荐方法）
            t = inputs['t']
            # 状态: h = w * t^2
            h = self.weight * t**2
            # 导数: dh/dt = 2 * w * t
            dh_dt = 2 * self.weight * t
            return {'state': h, 'derivative': dh_dt}
            
        else:
            raise ValueError(f"未知模式: {self.mode}")


class GridTimeModel(torch.nn.Module):
    """网格形式的时间相关模型，用于测试网格梯度流"""
    
    def __init__(self, mode='direct', grid_size=(8, 8)):
        super().__init__()
        self.mode = mode
        self.grid_size = grid_size
        # 创建一个简单的参数，使模型可训练
        self.weight = torch.nn.Parameter(torch.ones(1))
    
    def forward(self, inputs, coords=None):
        """
        参数:
            inputs: 包含't'时间的字典
            coords: 可选的坐标字典
        
        返回:
            根据mode返回不同格式的输出
        """
        batch_size = inputs['t'].shape[0]
        H, W = self.grid_size
        
        if self.mode == 'direct':
            # 直接使时间影响网格输出
            t = inputs['t']  # [B, 1]
            # 创建一个简单的时间依赖网格: h = w * t^2 * (空间变化)
            h = torch.zeros((batch_size, 1, H, W), device=t.device)
            
            # 添加空间变化
            y_grid, x_grid = torch.meshgrid(
                torch.linspace(0, 1, H, device=t.device),
                torch.linspace(0, 1, W, device=t.device),
                indexing='ij'
            )
            
            spatial_factor = torch.sin(2 * np.pi * x_grid) * torch.cos(2 * np.pi * y_grid)
            spatial_factor = spatial_factor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
            # 组合时间和空间因子
            for i in range(batch_size):
                h[i, 0] = self.weight * t[i]**2 * spatial_factor[0, 0]
                
            return {'pred': h}
            
        elif self.mode == 'dual_output':
            # 直接预测状态和导数（推荐方法）
            t = inputs['t']  # [B, 1]
            
            # 创建网格
            y_grid, x_grid = torch.meshgrid(
                torch.linspace(0, 1, H, device=t.device),
                torch.linspace(0, 1, W, device=t.device),
                indexing='ij'
            )
            
            spatial_factor = torch.sin(2 * np.pi * x_grid) * torch.cos(2 * np.pi * y_grid)
            spatial_factor = spatial_factor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
            # 初始化输出
            h = torch.zeros((batch_size, 1, H, W), device=t.device)
            dh_dt = torch.zeros((batch_size, 1, H, W), device=t.device)
            
            # 填充输出
            for i in range(batch_size):
                h[i, 0] = self.weight * t[i]**2 * spatial_factor[0, 0]
                dh_dt[i, 0] = 2 * self.weight * t[i] * spatial_factor[0, 0]
                
            return {'state': h, 'derivative': dh_dt}
            
        else:
            raise ValueError(f"未知模式: {self.mode}")


def test_grid_temporal_derivative_direct():
    """测试直接时间依赖下的网格时间导数计算"""
    
    # 创建简单的网格和时间输入
    batch_size = 2
    grid_size = (8, 8)
    t = torch.tensor([1.0, 2.0], requires_grad=True).view(batch_size, 1)
    
    # 创建模型并计算输出
    model = GridTimeModel(mode='direct', grid_size=grid_size)
    inputs = {'t': t}
    outputs = model(inputs)
    
    h_grid = outputs['pred']  # [B, 1, H, W]
    
    # 计算时间导数
    dh_dt = compute_grid_temporal_derivative(h_grid, t)
    
    # 检查导数是否计算成功（非None且形状正确）
    assert dh_dt is not None
    assert dh_dt.shape == h_grid.shape
    
    # 由于我们的模型是 h = w * t^2 * spatial_factor
    # 因此 dh/dt = 2 * w * t * spatial_factor
    # 检查每个批次的第一个点的导数是否约等于模型定义的值
    w = model.weight.item()
    
    # 获取空间因子样本值（仅检查第一个空间点）
    spatial_value = h_grid[0, 0, 0, 0].item() / (w * t[0].item()**2)
    
    # 验证第一个批次点的导数
    expected_value_batch0 = 2 * w * t[0].item() * spatial_value
    assert abs(dh_dt[0, 0, 0, 0].item() - expected_value_batch0) < 1e-5
    
    # 验证第二个批次点的导数
    expected_value_batch1 = 2 * w * t[1].item() * spatial_value
    assert abs(dh_dt[1, 0, 0, 0].item() - expected_value_batch1) < 1e-5


def test_grid_focused_pde_residual():
    """测试网格专注的PDE残差计算"""
    
    # 创建简单的网格和时间输入
    batch_size = 2
    grid_size = (8, 8)
    t = torch.tensor([1.0, 2.0], requires_grad=True).view(batch_size, 1)
    
    # 创建模型并计算输出
    model = GridTimeModel(mode='direct', grid_size=grid_size)
    inputs = {'t': t}
    outputs = model(inputs)
    
    h_grid = outputs['pred']  # [B, 1, H, W]
    
    # 设置物理参数（简化）
    physics_params = {
        'U': 0.001,
        'K_f': 1e-5,
        'm': 0.5,
        'n': 1.0,
        'K_d': 0.01,
        'dx': 1.0,
        'dy': 1.0,
        'precip': 1.0,
        'drainage_area_kwargs': {'temp': 0.1, 'num_iters': 5}
    }
    
    # 计算PDE残差
    residual_loss = compute_pde_residual_grid_focused(h_grid, t, physics_params)
    
    # 检查是否成功计算（结果应是标量损失且有限）
    assert residual_loss is not None
    assert residual_loss.ndim == 0  # 标量
    assert torch.isfinite(residual_loss)
    
    # 验证梯度是否可以传播
    residual_loss.backward()
    
    # 检查模型权重是否有梯度
    assert model.weight.grad is not None


def test_dual_output_pde_residual():
    """测试dual_output模式下的PDE残差计算"""
    
    # 创建简单的网格和时间输入
    batch_size = 2
    grid_size = (8, 8)
    t = torch.tensor([1.0, 2.0], requires_grad=True).view(batch_size, 1)
    
    # 创建模型并计算输出
    model = GridTimeModel(mode='dual_output', grid_size=grid_size)
    inputs = {'t': t}
    outputs = model(inputs)
    
    # 设置物理参数（简化）
    physics_params = {
        'U': 0.001,
        'K_f': 1e-5,
        'm': 0.5,
        'n': 1.0,
        'K_d': 0.01,
        'dx': 1.0,
        'dy': 1.0,
        'precip': 1.0,
        'drainage_area_kwargs': {'temp': 0.1, 'num_iters': 5}
    }
    
    # 计算PDE残差
    residual_loss = compute_pde_residual_dual_output(outputs, physics_params)
    
    # 检查是否成功计算（结果应是标量损失且有限）
    assert residual_loss is not None
    assert residual_loss.ndim == 0  # 标量
    assert torch.isfinite(residual_loss)
    
    # 验证梯度是否可以传播
    residual_loss.backward()
    
    # 检查模型权重是否有梯度
    assert model.weight.grad is not None


def test_total_loss_with_dual_output():
    """测试使用dual_output模式的物理损失进行总损失计算"""
    
    # 创建简单的网格和时间输入
    batch_size = 2
    grid_size = (8, 8)
    t = torch.tensor([1.0, 2.0], requires_grad=True).view(batch_size, 1)
    
    # 创建模型并计算输出
    model = GridTimeModel(mode='dual_output', grid_size=grid_size)
    inputs = {'t': t}
    outputs = model(inputs)
    
    # 设置物理参数（简化）
    physics_params = {
        'U': 0.001,
        'K_f': 1e-5,
        'm': 0.5,
        'n': 1.0,
        'K_d': 0.01,
        'dx': 1.0,
        'dy': 1.0,
        'precip': 1.0,
        'drainage_area_kwargs': {'temp': 0.1, 'num_iters': 5}
    }
    
    # 计算物理残差损失
    physics_loss = compute_pde_residual_dual_output(outputs, physics_params)
    
    # 创建虚拟目标数据
    target_data = torch.zeros_like(outputs['state'])
    
    # 设置损失权重
    loss_weights = {
        'data': 1.0,
        'physics': 0.1,
        'smoothness': 0.01,
        'conservation': 0.0
    }
    
    # 计算总损失
    total_loss, loss_dict = compute_total_loss(
        data_pred=outputs['state'],
        final_topo=target_data,
        physics_loss_value=physics_loss,
        physics_params=physics_params,
        loss_weights=loss_weights
    )
    
    # 检查总损失是否计算成功
    assert total_loss is not None
    assert torch.isfinite(total_loss)
    
    # 验证总损失包含所有组件
    assert 'data_loss' in loss_dict
    assert 'physics_loss' in loss_dict
    assert 'smoothness_loss' in loss_dict
    assert 'total_loss' in loss_dict
    
    # 验证梯度是否可以传播
    total_loss.backward()
    
    # 检查模型权重是否有梯度
    assert model.weight.grad is not None


def test_compare_gradient_methods():
    """比较不同梯度计算方法的一致性"""
    
    # 创建简单的网格和时间输入
    batch_size = 2
    grid_size = (8, 8)
    t = torch.tensor([1.0, 2.0], requires_grad=True).view(batch_size, 1)
    
    # 创建两种模式的模型
    model_direct = GridTimeModel(mode='direct', grid_size=grid_size)
    model_dual = GridTimeModel(mode='dual_output', grid_size=grid_size)
    
    # 复制权重以确保两个模型参数一致
    model_dual.weight.data.copy_(model_direct.weight.data)
    
    # 前向传播
    inputs = {'t': t}
    outputs_direct = model_direct(inputs)
    outputs_dual = model_dual(inputs)
    
    # 计算autograd方法的时间导数
    h_grid = outputs_direct['pred']
    dh_dt_autograd = compute_grid_temporal_derivative(h_grid, t)
    
    # 获取dual_output方法的时间导数
    dh_dt_dual = outputs_dual['derivative']
    
    # 比较两种方法计算的导数值
    # 检查两者是否接近
    diff = torch.abs(dh_dt_autograd - dh_dt_dual).mean().item()
    assert diff < 1e-5, f"导数计算方法差异过大: {diff}"


if __name__ == "__main__":
    # 运行所有测试
    pytest.main(["-xvs", __file__])
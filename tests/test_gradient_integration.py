import sys
import os
import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trainer import PINNTrainer
from src.losses import compute_pde_residual_dual_output


class SimpleGridPINN(torch.nn.Module):
    """用于测试的简化地形演化PINN模型"""
    
    def __init__(self, hidden_size=16, grid_size=(8, 8), predict_derivatives=True):
        super().__init__()
        self.grid_size = grid_size
        self.predict_derivatives = predict_derivatives
        
        # 一个简单的MLP编码器将时间编码到隐藏表示
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
        )
        
        # 解码器将隐藏表示转换为网格
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * 2),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size * 2, np.prod(grid_size)),
        )
        
        # 可选的导数预测器
        if predict_derivatives:
            self.derivative_decoder = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size * 2),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_size * 2, np.prod(grid_size)),
            )
    
    def forward(self, inputs, coords=None):
        """前向传播函数
        
        Args:
            inputs: 包含't'时间的字典
            coords: 可选的坐标点(未使用)
            
        Returns:
            包含'pred'或'state'/'derivative'的字典
        """
        # 获取时间输入
        t = inputs.get('t')
        if t is None:
            raise ValueError("输入必须包含't'键")
            
        batch_size = t.shape[0]
        H, W = self.grid_size
        
        # 编码时间
        encoded = self.encoder(t)
        
        # 解码为地形网格
        grid_flat = self.decoder(encoded)
        grid = grid_flat.view(batch_size, 1, H, W)
        
        if self.predict_derivatives:
            # 预测时间导数
            derivative_flat = self.derivative_decoder(encoded)
            derivative = derivative_flat.view(batch_size, 1, H, W)
            
            return {
                'state': grid,
                'derivative': derivative,
                'pred': grid,  # 兼容两种模式
            }
        else:
            return {'pred': grid}


class SimpleSyntheticDataset(Dataset):
    """用于测试的简单合成数据集，生成地形随时间的变化"""
    
    def __init__(self, num_samples=100, grid_size=(8, 8), t_range=(0, 10), seed=42):
        """初始化数据集
        
        Args:
            num_samples: 样本数量
            grid_size: 网格大小 (H, W)
            t_range: 时间范围 (min, max)
            seed: 随机种子
        """
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.t_range = t_range
        
        # 设置随机种子以确保可重复性
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 生成时间点
        self.time_points = torch.linspace(t_range[0], t_range[1], num_samples)
        
        # 生成初始条件 (简单的高斯山丘)
        H, W = grid_size
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing='ij'
        )
        
        # 创建初始高斯地形
        distance = torch.sqrt(x_grid**2 + y_grid**2)
        self.initial_dem = torch.exp(-5 * distance**2)
        
        # 模拟简化的地形演化 (侵蚀与抬升)
        self.evolution_rate = 0.1  # 演化速率
        self.uplift_rate = 0.05   # 抬升速率
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """获取单个样本
        
        Returns:
            包含数据和物理批次的字典
        """
        t = self.time_points[idx]
        
        # 模拟随时间演化的地形 (简化模型)
        # 通过指数衰减模拟侵蚀，并加上线性抬升
        H, W = self.grid_size
        evolved_dem = self.initial_dem * torch.exp(-self.evolution_rate * t) + self.uplift_rate * t
        
        # 整理成批次格式
        t_tensor = t.view(1, 1)  # [B=1, 1]
        dem_tensor = evolved_dem.unsqueeze(0).unsqueeze(0)  # [B=1, C=1, H, W]
        
        # 数据批次
        data_batch = {
            'inputs': {'t': t_tensor},
            'targets': {'h': dem_tensor}
        }
        
        # 物理批次 (在这个简化测试中，我们只需要时间)
        physics_batch = {
            'coords': {'t': t_tensor}
        }
        
        return {'data': data_batch, 'physics': physics_batch}


def test_trainer_with_dual_output():
    """测试trainer使用dual_output模式的端到端过程"""
    
    # 创建合成数据集
    dataset = SimpleSyntheticDataset(num_samples=20, grid_size=(8, 8))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 创建模型
    model = SimpleGridPINN(hidden_size=16, grid_size=(8, 8), predict_derivatives=True)
    
    # 初始化训练器
    trainer = PINNTrainer(
        model=model,
        lr=0.001,
        device='cpu',
        save_path=None,
        pde_residual_mode='dual_output', # 使用双输出模式
    )
    
    # 物理参数
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
    
    # 设置损失权重
    loss_weights = {
        'data': 1.0,
        'physics': 0.1,
        'smoothness': 0.01,
        'conservation': 0.0
    }
    
    # 训练单个epoch并验证Loss计算
    for batch in dataloader:
        loss_dict = trainer.train_epoch(
            batch['data'],
            batch['physics'],
            physics_params,
            loss_weights
        )
        
        # 检查损失是否合理
        assert 'total_loss' in loss_dict
        assert np.isfinite(loss_dict['total_loss'])
        assert loss_dict['total_loss'] > 0  # 确保损失不为零
        
        # 只运行一个批次来验证流程
        break


def test_gradient_backprop():
    """测试梯度反向传播过程"""
    
    # 创建模型和输入
    model = SimpleGridPINN(predict_derivatives=True)
    
    # 确保模型处于训练模式
    model.train()
    
    # 创建输入
    batch_size = 2
    t = torch.tensor([[1.0], [2.0]], requires_grad=True)
    inputs = {'t': t}
    
    # 前向传播
    outputs = model(inputs)
    
    # 检查输出是否包含必要的键
    assert 'state' in outputs
    assert 'derivative' in outputs
    
    # 物理参数
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
    
    # 计算物理残差
    physics_loss = compute_pde_residual_dual_output(outputs, physics_params)
    
    # 检查损失是否合理
    assert torch.isfinite(physics_loss)
    assert physics_loss.requires_grad
    
    # 反向传播
    physics_loss.backward()
    
    # 检查所有参数是否有梯度
    for name, param in model.named_parameters():
        assert param.grad is not None, f"参数 {name} 没有梯度"
        assert torch.isfinite(param.grad).all(), f"参数 {name} 的梯度包含NaN/Inf"


def test_compare_residual_modes():
    """比较不同PDE残差计算模式"""
    
    # 创建能同时支持两种模式的模型
    model = SimpleGridPINN(predict_derivatives=True)
    
    # 创建输入
    batch_size = 2
    t = torch.tensor([[1.0], [2.0]], requires_grad=True)
    inputs = {'t': t}
    
    # 前向传播
    outputs = model(inputs)
    
    # 物理参数
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
    
    # 初始化两种训练器
    trainer_grid = PINNTrainer(
        model=model,
        pde_residual_mode='grid_focused',
        device='cpu'
    )
    
    trainer_dual = PINNTrainer(
        model=model,
        pde_residual_mode='dual_output',
        device='cpu'
    )
    
    # 使用两种模式计算物理损失
    physics_loss_grid = trainer_grid._compute_physics_loss(
        outputs,
        {'t': t},
        physics_params
    )
    
    physics_loss_dual = trainer_dual._compute_physics_loss(
        outputs,
        {'t': t},
        physics_params
    )
    
    # 两种方法都应该能成功计算（虽然值可能不完全相同）
    assert torch.isfinite(physics_loss_grid)
    assert torch.isfinite(physics_loss_dual)
    
    # 记录每种方法的梯度影响
    # 重置梯度
    model.zero_grad()
    physics_loss_grid.backward(retain_graph=True)
    grid_grads = {name: param.grad.clone() for name, param in model.named_parameters()}
    
    model.zero_grad()
    physics_loss_dual.backward()
    dual_grads = {name: param.grad.clone() for name, param in model.named_parameters()}
    
    # 验证两种方法都产生了梯度
    for name in grid_grads:
        assert torch.isfinite(grid_grads[name]).all(), f"grid_focused模式在参数 {name} 上产生了NaN梯度"
        assert torch.isfinite(dual_grads[name]).all(), f"dual_output模式在参数 {name} 上产生了NaN梯度"
    
    # 打印总结信息
    print("\n比较不同PDE残差计算模式:")
    print(f"grid_focused 损失值: {physics_loss_grid.item():.6f}")
    print(f"dual_output 损失值: {physics_loss_dual.item():.6f}")
    

if __name__ == "__main__":
    # 运行所有测试
    pytest.main(["-xvs", __file__])
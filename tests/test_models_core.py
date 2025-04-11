import torch
import pytest
import logging

# Import necessary model classes
from src.models import MLP_PINN, FastscapePINN, TimeDerivativePINN

# Basic logging setup for tests
logging.basicConfig(level=logging.DEBUG)

# --- Fixtures (Consider moving complex ones to conftest.py) ---

@pytest.fixture
def dummy_config_mlp():
    """Basic config for MLP_PINN."""
    return {
        'model': {
            'name': 'MLP_PINN',
            'input_dim': 3, # x, y, t
            'output_dim': 1,
            'hidden_layers': 4,
            'hidden_neurons': 64
        }
    }

@pytest.fixture
def dummy_config_fastscape():
    """Basic config for FastscapePINN."""
    return {
        'model': {
            'name': 'FastscapePINN',
            'input_dim': 3, # x, y, t for MLP part
            'output_dim': 1,
            'hidden_dim': 64,
            'num_layers': 4,
            'grid_height': 16,
            'grid_width': 16,
            'num_param_channels': 3 # Example: K, D, U
        }
    }

@pytest.fixture
def dummy_coord_input():
    """Dummy input for predict_coords mode."""
    N = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return {
        'x': torch.rand(N, 1, device=device, requires_grad=True),
        'y': torch.rand(N, 1, device=device, requires_grad=True),
        't': torch.rand(N, 1, device=device, requires_grad=True)
        # Add dummy params if model input_dim > 3
        # 'k': torch.rand(N, 1, device=device),
        # 'u': torch.rand(N, 1, device=device),
    }

@pytest.fixture
def dummy_state_input(dummy_config_fastscape):
    """Dummy input for predict_state mode."""
    cfg = dummy_config_fastscape['model']
    B, H, W = 2, cfg['grid_height'], cfg['grid_width']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    initial_state = torch.rand(B, 1, H, W, device=device)
    params = {
        'K': torch.rand(B, 1, H, W, device=device) * 1e-5, # Example spatial K
        'D': torch.tensor(0.01, device=device), # Example scalar D
        'U': torch.rand(B, device=device) * 0.001 # Example batch scalar U
    }
    t_target = torch.tensor(500.0, device=device)
    return {'initial_state': initial_state, 'params': params, 't_target': t_target}


# --- Test Cases ---

# 1. Test TimeDerivativePINN Interface (using a dummy subclass)
class DummyDualOutputModel(TimeDerivativePINN):
    def __init__(self):
        super().__init__()
        # Dummy layers just to make it runnable
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        # Dummy forward, doesn't actually use output_state/derivative flags here
        # In real models, forward would check these flags
        dummy_output = self.linear(torch.rand(x.shape[0], 10, device=x.device))
        outputs = {}
        if self.output_state:
            outputs['state'] = dummy_output
        if self.output_derivative:
            outputs['derivative'] = dummy_output * 0.1 # Dummy derivative
        return outputs if len(outputs) > 1 else next(iter(outputs.values()))

def test_time_derivative_pinn_interface():
    """Tests the get/set output mode methods of the base class."""
    model = DummyDualOutputModel()
    assert model.get_output_mode() == ['state', 'derivative']

    model.set_output_mode(state=True, derivative=False)
    assert model.get_output_mode() == ['state']
    # Test forward call (though dummy doesn't use flags internally)
    # output = model(torch.rand(5,1))
    # assert 'state' in output and 'derivative' not in output # This check depends on actual model impl

    model.set_output_mode(state=False, derivative=True)
    assert model.get_output_mode() == ['derivative']

    model.set_output_mode(state=True, derivative=True)
    assert model.get_output_mode() == ['state', 'derivative']

# 2. Test MLP_PINN
def test_mlp_pinn_init(dummy_config_mlp):
    """Tests MLP_PINN initialization."""
    cfg = dummy_config_mlp['model']
    model = MLP_PINN(
        input_dim=cfg['input_dim'],
        output_dim=cfg['output_dim'],
        hidden_layers=cfg['hidden_layers'],
        hidden_neurons=cfg['hidden_neurons']
    )
    assert isinstance(model, MLP_PINN)
    # Add more checks: number of layers, activation functions etc.

def test_mlp_pinn_forward_coords(dummy_config_mlp, dummy_coord_input):
    """Tests MLP_PINN forward pass in predict_coords mode."""
    cfg = dummy_config_mlp['model']
    model = MLP_PINN(input_dim=cfg['input_dim'], output_dim=cfg['output_dim'])
    # Set model to output only state to match the original test assertion expectation
    model.set_output_mode(state=True, derivative=False)
    output = model(dummy_coord_input, mode='predict_coords')
    # Now that the model is set to single output, assert it's a tensor
    assert isinstance(output, torch.Tensor)
    assert output.shape == (len(dummy_coord_input['x']), cfg['output_dim'])
    assert output.requires_grad # Check gradient connection

# TODO: Add test for MLP_PINN forward pass in predict_state mode
# def test_mlp_pinn_forward_state(dummy_config_mlp, dummy_state_input): ...

# 3. Test FastscapePINN
def test_fastscape_pinn_init(dummy_config_fastscape):
    """Tests FastscapePINN initialization."""
    cfg = dummy_config_fastscape['model']
    model = FastscapePINN(
        input_dim=cfg['input_dim'],
        output_dim=cfg['output_dim'],
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        grid_height=cfg['grid_height'],
        grid_width=cfg['grid_width'],
        num_param_channels=cfg['num_param_channels']
    )
    assert isinstance(model, FastscapePINN)
    assert hasattr(model, 'mlp')
    assert hasattr(model, 'encoder')
    assert hasattr(model, 'decoder')

def test_fastscape_pinn_forward_coords(dummy_config_fastscape, dummy_coord_input):
    """Tests FastscapePINN forward pass in predict_coords mode (uses MLP)."""
    cfg = dummy_config_fastscape['model']
    model = FastscapePINN(input_dim=cfg['input_dim'], output_dim=cfg['output_dim'])
    output = model(dummy_coord_input, mode='predict_coords')
    assert isinstance(output, torch.Tensor)
    assert output.shape == (len(dummy_coord_input['x']), cfg['output_dim'])
    assert output.requires_grad

def test_fastscape_pinn_forward_state(dummy_config_fastscape, dummy_state_input):
    """Tests FastscapePINN forward pass in predict_state mode (uses CNN)."""
    cfg = dummy_config_fastscape['model']
    model = FastscapePINN(
        input_dim=cfg['input_dim'], output_dim=cfg['output_dim'],
        grid_height=cfg['grid_height'], grid_width=cfg['grid_width'],
        num_param_channels=cfg['num_param_channels']
    )
    output = model(dummy_state_input, mode='predict_state')
    assert isinstance(output, torch.Tensor)
    B, H, W = dummy_state_input['initial_state'].shape[0], cfg['grid_height'], cfg['grid_width']
    assert output.shape == (B, cfg['output_dim'], H, W)
    # Check requires_grad if initial_state or params require grad (depends on setup)
    # assert output.requires_grad

# TODO: Add tests for handling different parameter types/shapes in FastscapePINN predict_state
# TODO: Add tests for edge cases (e.g., missing parameters)

# 添加一个测试用例，检查TimeDerivativePINN的输出模式和预期是否一致
def test_time_derivative_pinn_output_mode():
    """测试TimeDerivativePINN基类的输出模式功能"""
    from src.models import TimeDerivativePINN
    
    # 创建一个实现了TimeDerivativePINN的测试类
    class TestPINN(TimeDerivativePINN):
        def __init__(self):
            super().__init__()
            self.output_state = True
            self.output_derivative = True
            
        def forward(self, x, mode='test'):
            # 简单的前向实现，根据输出模式返回结果
            outputs = {}
            if self.output_state:
                outputs['state'] = torch.ones(1, 1)
            if self.output_derivative:
                outputs['derivative'] = torch.zeros(1, 1)
            
            if len(outputs) == 1:
                return next(iter(outputs.values()))
            return outputs
    
    # 创建模型实例
    model = TestPINN()
    
    # 测试默认模式（两个输出）
    output = model(None)
    assert isinstance(output, dict), "默认模式应该返回字典"
    assert 'state' in output, "默认模式应该包含状态输出"
    assert 'derivative' in output, "默认模式应该包含导数输出"
    
    # 测试只输出状态
    model.set_output_mode(state=True, derivative=False)
    output = model(None)
    assert not isinstance(output, dict), "单一输出模式应该返回张量而非字典"
    # 验证与state相同
    assert torch.all(output == torch.ones(1, 1)), "单状态模式应该只返回状态张量"
    
    # 测试只输出导数
    model.set_output_mode(state=False, derivative=True)
    output = model(None)
    assert not isinstance(output, dict), "单一输出模式应该返回张量而非字典"
    # 验证与derivative相同
    assert torch.all(output == torch.zeros(1, 1)), "单导数模式应该只返回导数张量"
    
    # 测试get_output_mode
    model.set_output_mode(state=True, derivative=True)
    modes = model.get_output_mode()
    assert 'state' in modes, "modes应包含'state'"
    assert 'derivative' in modes, "modes应包含'derivative'"
    assert len(modes) == 2, "应该有两个模式"
    
    # 测试无输出（应该抛出错误）
    with pytest.raises(ValueError, match="至少需要一个输出模式为True"):
        model.set_output_mode(state=False, derivative=False)
        
    # 确保继续设置为有效状态
    model.set_output_mode(state=True, derivative=True)
import os
import sys # Added import
import logging
import random
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf # Import OmegaConf
import time
from logging.handlers import RotatingFileHandler

def setup_logging(log_level='INFO', log_file=None, log_to_console=True):
    """
    设置增强的日志系统，支持文件和控制台输出。
    
    Args:
        log_level: 日志级别，可以是 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        log_file: 日志文件路径，如果为 None 则不输出到文件
        log_to_console: 是否输出到控制台
        
    Returns:
        logging.Logger: 配置好的日志器
    """
    # 创建日志格式
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 获取根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除已有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加控制台处理器
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        root_logger.addHandler(console_handler)
    
    # 添加文件处理器
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        # 创建滚动文件处理器，限制文件大小和数量
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(log_format)
        root_logger.addHandler(file_handler)
    
    # 记录启动信息
    root_logger.info(f"日志系统初始化完成。级别: {log_level}, 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return root_logger


class ErrorHandler:
    """
    错误处理器，提供更精细的错误处理和恢复策略。
    
    使用示例:
    ```
    # 创建错误处理器
    error_handler = ErrorHandler(max_retries=3)
    
    # 使用装饰器捕获和处理函数中的错误
    @error_handler.catch_and_handle(retry_on=[ValueError], ignore=[KeyError])
    def process_data(data):
        # 处理数据...
    
    # 或者使用上下文管理器
    with error_handler.handling_context(reraise=False):
        # 执行可能出错的代码...
    ```
    """
    
    def __init__(self, max_retries=3, default_reraise=True, default_retry_on=None):
        """
        初始化错误处理器。
        
        Args:
            max_retries: 默认的最大重试次数
            default_reraise: 默认是否重新抛出无法处理的错误
            default_retry_on: 默认重试的错误类型列表
        """
        self.max_retries = max_retries
        self.default_reraise = default_reraise
        self.default_retry_on = default_retry_on or []
        
    def catch_and_handle(self, retry_on=None, ignore=None, reraise=None, max_retries=None):
        """
        捕获并处理函数中的错误的装饰器。
        
        Args:
            retry_on: 需要重试的错误类型列表
            ignore: 需要忽略的错误类型列表
            reraise: 是否重新抛出无法处理的错误
            max_retries: 最大重试次数
            
        Returns:
            function: 装饰器函数
        """
        retry_on = retry_on or self.default_retry_on
        ignore = ignore or []
        reraise = self.default_reraise if reraise is None else reraise
        max_retries = self.max_retries if max_retries is None else max_retries
        
        def decorator(func):
            import functools
            import logging
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                retries = 0
                
                while True:
                    try:
                        return func(*args, **kwargs)
                    except tuple(ignore) as e:
                        logging.info(f"忽略错误 {type(e).__name__}: {str(e)}")
                        return None
                    except tuple(retry_on) as e:
                        retries += 1
                        if retries <= max_retries:
                            logging.warning(
                                f"捕获到可重试错误 {type(e).__name__}: {str(e)}。"
                                f"重试 {retries}/{max_retries}..."
                            )
                            continue
                        if reraise:
                            logging.error(
                                f"重试次数已用尽 ({max_retries})。重新抛出错误: {type(e).__name__}"
                            )
                            raise
                        logging.error(
                            f"重试次数已用尽 ({max_retries})。返回 None。错误: {type(e).__name__}: {str(e)}"
                        )
                        return None
                    except Exception as e:
                        if reraise:
                            logging.error(f"捕获到未处理的错误 {type(e).__name__}: {str(e)}。重新抛出。")
                            raise
                        logging.error(f"捕获到未处理的错误 {type(e).__name__}: {str(e)}。返回 None。")
                        return None
            
            return wrapper
        
        return decorator
    
    def handling_context(self, retry_on=None, ignore=None, reraise=None, max_retries=None):
        """
        创建一个用于处理错误的上下文管理器。
        
        Args:
            retry_on: 需要重试的错误类型列表
            ignore: 需要忽略的错误类型列表
            reraise: 是否重新抛出无法处理的错误
            max_retries: 最大重试次数
            
        Returns:
            ErrorHandlingContext: 错误处理上下文
        """
        return ErrorHandlingContext(
            retry_on=retry_on or self.default_retry_on,
            ignore=ignore or [],
            reraise=self.default_reraise if reraise is None else reraise,
            max_retries=self.max_retries if max_retries is None else max_retries
        )


class ErrorHandlingContext:
    """错误处理上下文管理器"""
    
    def __init__(self, retry_on, ignore, reraise, max_retries):
        """
        初始化错误处理上下文。
        
        Args:
            retry_on: 需要重试的错误类型列表
            ignore: 需要忽略的错误类型列表
            reraise: 是否重新抛出无法处理的错误
            max_retries: 最大重试次数
        """
        self.retry_on = retry_on
        self.ignore = ignore
        self.reraise = reraise
        self.max_retries = max_retries
        self.retries = 0
        self.last_error = None
        
    def __enter__(self):
        """进入上下文"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，处理可能的错误"""
        import logging
        
        if exc_type is None:
            return True
            
        if issubclass(exc_type, tuple(self.ignore)):
            logging.info(f"忽略错误 {exc_type.__name__}: {str(exc_val)}")
            return True
            
        if issubclass(exc_type, tuple(self.retry_on)):
            self.retries += 1
            self.last_error = (exc_type, exc_val, exc_tb)
            
            if self.retries <= self.max_retries:
                logging.warning(
                    f"捕获到可重试错误 {exc_type.__name__}: {str(exc_val)}。"
                    f"重试 {self.retries}/{self.max_retries}..."
                )
                # 不抑制异常，让调用者进行重试
                return False
                
            if not self.reraise:
                logging.error(
                    f"重试次数已用尽 ({self.max_retries})。抑制错误: {exc_type.__name__}: {str(exc_val)}"
                )
                return True
                
            logging.error(
                f"重试次数已用尽 ({self.max_retries})。重新抛出错误: {exc_type.__name__}"
            )
            return False
            
        if not self.reraise:
            logging.error(f"捕获到未处理的错误 {exc_type.__name__}: {str(exc_val)}。抑制错误。")
            return True
            
        logging.error(f"捕获到未处理的错误 {exc_type.__name__}: {str(exc_val)}。重新抛出。")
        return False
        
    def should_retry(self):
        """检查是否应该重试"""
        return self.retries < self.max_retries

def get_device(config):
    """Gets the appropriate torch device based on config and availability."""
    if config.get('device', 'auto') == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config['device']

    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA specified but not available. Falling back to CPU.")
        device = "cpu"

    logging.info(f"Using device: {device}")
    return torch.device(device)

def set_seed(seed):
    """Sets random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) # for multi-GPU.
            # Ensure deterministic algorithms are used where possible
            # Note: some operations may still be non-deterministic
            # torch.use_deterministic_algorithms(True) # Use if needed, requires PyTorch 1.8+
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logging.info(f"Set random seed to {seed}")

def save_data_sample(data_dict, filepath):
    """Saves a data sample dictionary to a .pt file."""
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(data_dict, filepath)
        # logging.info(f"Successfully saved data sample to {filepath}") # Reduce log verbosity
    except Exception as e:
        logging.error(f"Error saving file {filepath}: {e}")

def load_config(config_path):
    """Loads a YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            # config = yaml.safe_load(f) # Replace yaml.safe_load
            # Use OmegaConf to load and resolve interpolations/calculations
            config = OmegaConf.load(f)
            # Optionally resolve interpolations immediately if needed,
            # though often resolution happens implicitly on access.
            # OmegaConf.resolve(config) # Uncomment if explicit resolution is required
        logging.info(f"Loaded configuration from {config_path} using OmegaConf")
        # Return the OmegaConf object (or convert back to dict if necessary, but OmegaConf object is often preferred)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading config file {config_path}: {e}")
        raise

def save_config(config, filepath):
    """Saves a configuration dictionary to a YAML file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logging.info(f"Configuration saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving config file {filepath}: {e}")

def prepare_parameter(param_value, target_shape=None, batch_size=None, device=None, dtype=None, param_name="unknown"):
    """
    统一处理不同形式的参数值，确保输出一致的形状和类型。
    可用于处理标量、张量或空间场参数。
    
    Args:
        param_value: 参数值（标量、张量或数组）
        target_shape: 目标空间形状 (H, W) 或 None
        batch_size: 批次大小或 None
        device: 目标设备或 None
        dtype: 目标数据类型或 None
        param_name: 参数名称，用于错误消息
        
    Returns:
        torch.Tensor: 处理后的参数，具有合适的形状和类型
        
    示例:
        K_field = prepare_parameter(K, (64, 64), batch_size=8, device='cuda')
        # 如果 K 是标量，返回形状为 [8, 1, 64, 64] 的张量
        # 如果 K 是形状为 [64, 64] 的张量，返回形状为 [8, 1, 64, 64] 的张量
        # 如果 K 是形状为 [8, 1, 64, 64] 的张量，直接返回
    """
    import torch
    import logging
    
    # 处理 None 值
    # 处理 None 值
    if param_value is None:
        if target_shape is not None and batch_size is not None and device is not None and dtype is not None:
            logging.warning(f"参数 '{param_name}' 为 None，返回形状为 {(batch_size, 1, *target_shape)} 的零张量")
            return torch.zeros((batch_size, 1, *target_shape), device=device, dtype=dtype)
        else:
            logging.warning(f"参数 '{param_name}' 为 None 且缺少形状/设备/类型信息，返回值为 None")
            return None
    
    # 获取设备参考
    if device is None and isinstance(param_value, torch.Tensor):
        device = param_value.device
    elif device is None:
        device = torch.device('cpu')
    
    # 获取数据类型参考
    if dtype is None and isinstance(param_value, torch.Tensor):
        dtype = param_value.dtype
    elif dtype is None:
        dtype = torch.float32
        
    # 处理标量值（数字）
    if isinstance(param_value, (int, float)):
        if target_shape is None or batch_size is None:
            # 标量值，无需扩展
            return torch.tensor(param_value, device=device, dtype=dtype)
        else:
            # 创建指定形状的全填充张量
            return torch.full((batch_size, 1, *target_shape), float(param_value), device=device, dtype=dtype)
    
    # 处理张量
    elif isinstance(param_value, torch.Tensor):
        # 确保张量在正确的设备和数据类型上
        param_tensor = param_value.to(device=device, dtype=dtype)
        
        # 如果不需要特定形状，直接返回转换后的张量
        if target_shape is None or batch_size is None:
            return param_tensor
            
        # 处理各种形状的张量
        if param_tensor.numel() == 1:
            # 标量张量，扩展到目标形状
            return torch.full((batch_size, 1, *target_shape), param_tensor.item(), device=device, dtype=dtype)
            
        elif param_tensor.ndim == 1 and param_tensor.shape[0] == batch_size:
            # 批次向量 [B]，扩展为 [B, 1, H, W]
            return param_tensor.view(batch_size, 1, 1, 1).expand(batch_size, 1, *target_shape)
        
        elif param_tensor.ndim == 2 and param_tensor.shape[0] == batch_size and param_tensor.shape[1] == 1:
             # 批次标量 [B, 1]，扩展为 [B, 1, H, W]
             return param_tensor.view(batch_size, 1, 1, 1).expand(batch_size, 1, *target_shape)
            
        elif param_tensor.ndim == 2 and param_tensor.shape == target_shape:
            # 空间场 [H, W]，扩展为 [B, 1, H, W]
            return param_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, *target_shape)
            
        elif param_tensor.ndim == 3:
            if param_tensor.shape[0] == batch_size and param_tensor.shape[1:] == target_shape:
                # 带批次的空间场 [B, H, W]，添加通道维度
                return param_tensor.unsqueeze(1)
            elif param_tensor.shape[-2:] == target_shape:
                # 其他 3D 张量，尝试广播
                return _try_expand_tensor(param_tensor, (batch_size, 1, *target_shape))
                
        elif param_tensor.ndim == 4:
            if param_tensor.shape == (batch_size, 1, *target_shape):
                # 已经是目标形状 [B, 1, H, W]
                return param_tensor
            else:
                # 其他 4D 张量，尝试广播
                return _try_expand_tensor(param_tensor, (batch_size, 1, *target_shape))
        
        # 尝试直接广播
        return _try_expand_tensor(param_tensor, (batch_size, 1, *target_shape))
    
    # 处理 NumPy 数组
    elif hasattr(param_value, '__array__') and callable(getattr(param_value, '__array__')):
        import numpy as np
        # 将 NumPy 数组转换为 PyTorch 张量
        return prepare_parameter(torch.tensor(np.asarray(param_value), device=device, dtype=dtype), 
                                 target_shape, batch_size, device, dtype, param_name)
    
    # 不支持的类型
    else:
        raise TypeError(f"参数 '{param_name}' 的类型 '{type(param_value)}' 不受支持，应为标量、张量或 NumPy 数组")


def _try_expand_tensor(tensor, target_shape):
    """
    尝试将张量扩展到目标形状，如果失败则引发有用的错误。
    
    Args:
        tensor: 输入张量
        target_shape: 目标形状元组
        
    Returns:
        扩展后的张量
    """
    try:
        return tensor.expand(target_shape)
    except RuntimeError:
        raise ValueError(f"无法将形状为 {tensor.shape} 的张量广播到目标形状 {target_shape}")


def standardize_coordinate_system(coords, domain_x=(0, 1), domain_y=(0, 1), 
                                  normalize=False, device=None, dtype=None):
    """
    标准化坐标系，确保不同模块使用一致的坐标表示。
    
    Args:
        coords: 坐标字典 {'x': x_tensor, 'y': y_tensor, ...} 或坐标元组 (x_tensor, y_tensor)
        domain_x: x轴的物理域 (x_min, x_max)
        domain_y: y轴的物理域 (y_min, y_max)
        normalize: 是否将物理坐标归一化到 [0,1] 范围
        device: 目标设备，如果为 None 则使用输入坐标的设备
        dtype: 目标数据类型，如果为 None 则使用输入坐标的类型
        
    Returns:
        dict: 标准化的坐标字典 {'x': x_tensor, 'y': y_tensor, ...}
    """
    import torch
    
    # 处理不同的输入形式
    if isinstance(coords, dict):
        x = coords.get('x')
        y = coords.get('y')
        extra_keys = {k: v for k, v in coords.items() if k not in ['x', 'y']}
        if x is None or y is None:
            raise ValueError("坐标字典必须包含 'x' 和 'y' 键")
    elif isinstance(coords, (list, tuple)) and len(coords) >= 2:
        x, y = coords[0], coords[1]
        extra_keys = {}
    else:
        raise ValueError("坐标必须是包含 'x'/'y' 键的字典或至少有两个元素的元组/列表")
    
    # 确定设备和数据类型
    if device is None and isinstance(x, torch.Tensor):
        device = x.device
    if dtype is None and isinstance(x, torch.Tensor):
        dtype = x.dtype
    
    # 将坐标转换为张量
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, device=device, dtype=dtype)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, device=device, dtype=dtype)
    
    # 确保坐标在正确的设备和数据类型上
    x = x.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=dtype)
    
    # 归一化坐标（如果需要）
    if normalize:
        x_min, x_max = domain_x
        y_min, y_max = domain_y
        x_norm = (x - x_min) / (x_max - x_min)
        y_norm = (y - y_min) / (y_max - y_min)
        result = {'x': x_norm, 'y': y_norm}
    else:
        result = {'x': x, 'y': y}
    
    # 添加额外的键
    for k, v in extra_keys.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device=device, dtype=dtype)
        else:
            result[k] = v
    
    return result

if __name__ == '__main__':
    # Example usage
    # Setup basic logging for standalone run info
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    setup_logging("temp_logs", "utils_test.log")
    logging.info("Testing utils...")

    # Test device selection
    dummy_config = {'device': 'auto'}
    device = get_device(dummy_config)
    logging.info(f"Device selected: {device}")

    # Test seed setting
    set_seed(42)
    logging.info(f"Random float after seed: {random.random()}")

    # Test config loading/saving (create dummy)
    dummy_cfg_path = "temp_logs/dummy_config.yaml"
    dummy_cfg_data = {'learning_rate': 0.001, 'model': {'type': 'MLP'}}
    save_config(dummy_cfg_data, dummy_cfg_path)
    loaded_cfg = load_config(dummy_cfg_path)
    logging.info(f"Loaded config: {loaded_cfg}")

    # Test data saving
    dummy_data = {'tensor': torch.randn(2, 2)}
    save_data_sample(dummy_data, "temp_logs/dummy_sample.pt")

    # Clean up
    import shutil
    if os.path.exists("temp_logs"):
        shutil.rmtree("temp_logs")
        logging.info("Cleaned up temp_logs directory.")

    logging.info("Utils testing done.")

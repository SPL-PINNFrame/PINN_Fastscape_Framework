"""
使用PIL库生成叠加图像：展示地形和汇水面积的叠加效果
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# 导入汇水面积计算函数
from src.physics import calculate_drainage_area_differentiable_optimized
try:
    from src.drainage_area_enhanced import calculate_drainage_area_enhanced
    HAS_ENHANCED_DRAINAGE = True
except ImportError:
    HAS_ENHANCED_DRAINAGE = False
    print("增强版汇水面积计算不可用。")

def create_test_dem(size=64, device='cpu', dtype=torch.float32):
    """
    创建测试用DEM，包含各种地形特征
    """
    # 创建坐标网格
    y, x = torch.meshgrid(
        torch.arange(size, dtype=dtype, device=device),
        torch.arange(size, dtype=dtype, device=device),
        indexing='ij'
    )
    
    # 创建倾斜地表
    dem = 100.0 - 0.5 * x - 0.2 * y
    
    # 添加一些噪声
    torch.manual_seed(42)  # 设置随机种子以确保可重复性
    noise = torch.randn(size, size, device=device, dtype=dtype) * 0.2
    dem = dem + noise
    
    # 添加洼地（坑）
    center_x, center_y = size // 3, size // 3
    radius_sq = (size // 10) ** 2
    dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
    depression_depth = 10.0
    dem = dem - depression_depth * torch.exp(-dist_sq / (2 * radius_sq)) * (dist_sq < radius_sq).float()
    
    # 添加山脊
    ridge_x = 2 * size // 3
    ridge_width = size // 20
    ridge_height = 5.0
    dem = dem + ridge_height * torch.exp(-(x - ridge_x) ** 2 / (2 * ridge_width ** 2))
    
    # 添加平坦区域
    flat_x_min, flat_x_max = 3 * size // 4, 7 * size // 8
    flat_y_min, flat_y_max = size // 8, size // 4
    flat_mask = ((x >= flat_x_min) & (x <= flat_x_max) & 
                 (y >= flat_y_min) & (y <= flat_y_max))
    flat_value = dem[flat_y_min, flat_x_min]
    dem = torch.where(flat_mask, flat_value, dem)
    
    # 重塑为(1, 1, size, size)以适应模型
    return dem.unsqueeze(0).unsqueeze(0)

def array_to_image(array, colormap='viridis'):
    """
    将numpy数组转换为PIL图像，应用颜色映射
    
    Args:
        array: 要转换的numpy数组
        colormap: 颜色映射名称 ('terrain', 'blues', 'viridis', 'reds')
        
    Returns:
        PIL.Image: 彩色图像
    """
    # 归一化数组到0-1范围
    min_val = array.min()
    max_val = array.max()
    if max_val > min_val:
        normalized = (array - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(array)
    
    # 转换为0-255范围的整数
    img_array = (normalized * 255).astype(np.uint8)
    
    # 创建灰度图像
    img = Image.fromarray(img_array, mode='L')
    
    # 应用颜色映射
    if colormap.lower() == 'terrain':
        # 地形颜色映射：从绿色（低）到棕色（中）到白色（高）
        img = img.convert('L')
        img_colored = Image.new('RGB', img.size)
        for y in range(img.height):
            for x in range(img.width):
                val = img.getpixel((x, y))
                if val < 85:  # 低海拔：绿色
                    r = int(0.4 * val)
                    g = int(0.6 * val + 100)
                    b = int(0.4 * val)
                elif val < 170:  # 中海拔：棕色
                    r = int(0.6 * val + 40)
                    g = int(0.5 * val + 40)
                    b = int(0.4 * val)
                else:  # 高海拔：白色
                    r = int(0.8 * val + 50)
                    g = int(0.8 * val + 50)
                    b = int(0.8 * val + 50)
                img_colored.putpixel((x, y), (r, g, b))
        return img_colored
    
    elif colormap.lower() == 'blues':
        # 蓝色映射：从白色（低）到深蓝色（高）
        img = img.convert('L')
        img_colored = Image.new('RGB', img.size)
        for y in range(img.height):
            for x in range(img.width):
                val = img.getpixel((x, y))
                r = int(255 - 0.7 * val)
                g = int(255 - 0.5 * val)
                b = int(255 - 0.1 * val)
                img_colored.putpixel((x, y), (r, g, b))
        return img_colored
    
    elif colormap.lower() == 'reds':
        # 红色映射：从白色（低）到深红色（高）
        img = img.convert('L')
        img_colored = Image.new('RGB', img.size)
        for y in range(img.height):
            for x in range(img.width):
                val = img.getpixel((x, y))
                r = int(255 - 0.1 * val)
                g = int(255 - 0.7 * val)
                b = int(255 - 0.7 * val)
                img_colored.putpixel((x, y), (r, g, b))
        return img_colored
    
    else:  # viridis (默认)
        # 简化的viridis颜色映射
        img = img.convert('L')
        img_colored = Image.new('RGB', img.size)
        for y in range(img.height):
            for x in range(img.width):
                val = img.getpixel((x, y))
                if val < 85:  # 低值：深蓝色到紫色
                    r = int(0.4 * val)
                    g = int(0.2 * val)
                    b = int(0.5 * val + 100)
                elif val < 170:  # 中值：紫色到绿色
                    r = int(0.3 * val)
                    g = int(0.5 * val + 40)
                    b = int(0.5 * val + 40)
                else:  # 高值：绿色到黄色
                    r = int(0.6 * val + 40)
                    g = int(0.7 * val + 40)
                    b = int(0.2 * val)
                img_colored.putpixel((x, y), (r, g, b))
        return img_colored

def log_transform(array, epsilon=1e-10):
    """
    对数变换：log(1 + x)
    """
    return np.log1p(array + epsilon)

def create_overlay_image(dem_array, drainage_array, filename, log_scale=True, alpha=0.7):
    """
    创建地形和汇水面积的叠加图像
    
    Args:
        dem_array: 地形高程数据数组
        drainage_array: 汇水面积数据数组
        filename: 输出文件名
        log_scale: 是否对汇水面积使用对数尺度
        alpha: 汇水面积图层的透明度
    """
    try:
        # 转换地形为图像
        dem_img = array_to_image(dem_array, colormap='terrain')
        
        # 转换汇水面积为图像
        if log_scale:
            drainage_log = log_transform(drainage_array)
            drainage_img = array_to_image(drainage_log, colormap='blues')
        else:
            drainage_img = array_to_image(drainage_array, colormap='blues')
        
        # 创建叠加图像
        overlay_img = Image.blend(dem_img, drainage_img, alpha=alpha)
        
        # 保存图像
        overlay_img.save(filename)
        print(f"已保存叠加图像到 {filename}")
        
        # 同时保存单独的图像
        dem_img.save(f"{os.path.splitext(filename)[0]}_dem.png")
        drainage_img.save(f"{os.path.splitext(filename)[0]}_drainage.png")
        
    except Exception as e:
        print(f"创建叠加图像失败: {e}")

def main():
    """主函数：计算汇水面积并生成可视化结果"""
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建测试DEM
    size = 64
    print(f"创建大小为{size}x{size}的测试DEM...")
    dem = create_test_dem(size=size, device=device)
    
    # 参数
    dx = 10.0
    dy = 10.0
    precip = 1.0
    
    # 使用原始方法计算汇水面积
    print("\n--- 运行原始方法 ---")
    start_time = time.time()
    original_params = {
        'temp': 0.01,
        'num_iters': 50,
        'verbose': True
    }
    original_result = calculate_drainage_area_differentiable_optimized(
        dem, dx=dx, dy=dy, precip=precip, **original_params
    )
    original_time = time.time() - start_time
    print(f"原始方法完成，耗时 {original_time:.3f} 秒")
    
    # 保存原始方法结果
    dem_np = dem[0, 0].cpu().numpy()
    original_np = original_result[0, 0].cpu().numpy()
    
    create_overlay_image(
        dem_np, original_np,
        os.path.join(output_dir, 'original_overlay.png'),
        log_scale=True
    )
    
    # 使用增强版方法计算汇水面积（如果可用）
    if HAS_ENHANCED_DRAINAGE:
        print("\n--- 运行增强版方法 ---")
        start_time = time.time()
        enhanced_params = {
            'initial_temp': 0.1,
            'end_temp': 1e-3,
            'annealing_factor': 0.99,
            'max_iters': 20,
            'lambda_dir': 1.0,
            'convergence_threshold': 1e-3,
            'special_depression_handling': True,
            'verbose': True,
            'stable_mode': True,
            'flat_handling': 'uniform',  # 使用均匀分布处理平坦区域
            'clamp_max_value': 1e5  # 使用较小的最大值以提高稳定性
        }
        enhanced_result = calculate_drainage_area_enhanced(
            dem, dx=dx, dy=dy, precip=precip, **enhanced_params
        )
        enhanced_time = time.time() - start_time
        print(f"增强版方法完成，耗时 {enhanced_time:.3f} 秒")
        
        # 保存增强版方法结果
        enhanced_np = enhanced_result[0, 0].cpu().numpy()
        
        create_overlay_image(
            dem_np, enhanced_np,
            os.path.join(output_dir, 'enhanced_overlay.png'),
            log_scale=True
        )
        
        # 计算统计数据
        abs_diff = torch.abs(enhanced_result - original_result)
        rel_diff = abs_diff / (original_result + 1e-10)
        
        max_abs_diff = torch.max(abs_diff).item()
        mean_abs_diff = torch.mean(abs_diff).item()
        max_rel_diff = torch.max(rel_diff).item()
        mean_rel_diff = torch.mean(rel_diff).item()
        
        print("\n--- 比较统计 ---")
        print(f"最大绝对差异: {max_abs_diff:.4f}")
        print(f"平均绝对差异: {mean_abs_diff:.4f}")
        print(f"最大相对差异: {max_rel_diff:.4f}")
        print(f"平均相对差异: {mean_rel_diff:.4f}")
        
        # 检查质量守恒
        total_precip = torch.sum(torch.ones_like(dem) * precip * dx * dy).item()
        total_original = torch.sum(original_result).item()
        total_enhanced = torch.sum(enhanced_result).item()
        
        print("\n--- 质量守恒 ---")
        print(f"总降水量: {total_precip:.4f}")
        print(f"原始方法总量: {total_original:.4f}")
        print(f"增强版方法总量: {total_enhanced:.4f}")
        print(f"原始方法质量误差: {(total_original - total_precip) / total_precip:.4%}")
        print(f"增强版方法质量误差: {(total_enhanced - total_precip) / total_precip:.4%}")
        
        # 保存差异图像
        abs_diff_np = abs_diff[0, 0].cpu().numpy()
        diff_img = array_to_image(abs_diff_np, colormap='reds')
        diff_img.save(os.path.join(output_dir, 'difference.png'))
    
    print("\n完成!")
    print(f"所有图像已保存到目录: {output_dir}")

if __name__ == "__main__":
    main()

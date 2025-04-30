"""
创建单高斯峰地形，使用Fastscape进行侵蚀，并计算汇水面积
"""

import os
import sys
import time
import numpy as np
import torch
from PIL import Image

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# 导入汇水面积计算函数
try:
    from src.drainage_area_enhanced import calculate_drainage_area_enhanced
    HAS_ENHANCED_DRAINAGE = True
except ImportError:
    HAS_ENHANCED_DRAINAGE = False
    print("增强版汇水面积计算不可用。")

def create_gaussian_peak(size=256, amplitude=1000.0, sigma=40.0, device='cpu', dtype=torch.float32):
    """
    创建单高斯峰地形

    Args:
        size: 网格大小
        amplitude: 高斯峰振幅
        sigma: 高斯峰标准差
        device: 计算设备
        dtype: 数据类型

    Returns:
        torch.Tensor: 高斯峰地形，形状为(1, 1, size, size)
    """
    # 创建坐标网格
    y, x = torch.meshgrid(
        torch.arange(size, dtype=dtype, device=device),
        torch.arange(size, dtype=dtype, device=device),
        indexing='ij'
    )

    # 计算中心点
    center_x, center_y = size // 2, size // 2

    # 计算到中心点的距离
    dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2

    # 创建高斯峰
    gaussian = amplitude * torch.exp(-dist_sq / (2 * sigma ** 2))

    # 添加基础高度，确保最低点不为负
    base_height = 100.0
    dem = gaussian + base_height

    # 重塑为(1, 1, size, size)以适应模型
    return dem.unsqueeze(0).unsqueeze(0)

def apply_stream_power_erosion(dem, dx, dy, K=1e-5, m=0.5, n=1.0, dt=1000, num_steps=100,
                              precip=1.0, device='cpu', use_enhanced=True):
    """
    应用流水侵蚀模型（Stream Power Law）侵蚀地形

    Args:
        dem: 初始地形，形状为(1, 1, H, W)
        dx: x方向网格间距
        dy: y方向网格间距
        K: 侵蚀系数
        m: 汇水面积指数
        n: 坡度指数
        dt: 时间步长
        num_steps: 时间步数
        precip: 降水率
        device: 计算设备
        use_enhanced: 是否使用增强版汇水面积计算

    Returns:
        torch.Tensor: 侵蚀后的地形，形状为(1, 1, H, W)
    """
    # 导入必要的函数
    from src.physics import calculate_slope_magnitude

    # 如果使用增强版汇水面积计算，但它不可用，则回退到原始方法
    if use_enhanced and not HAS_ENHANCED_DRAINAGE:
        use_enhanced = False
        print("增强版汇水面积计算不可用，使用原始方法。")

    # 选择汇水面积计算函数
    if use_enhanced:
        from src.drainage_area_enhanced import calculate_drainage_area_enhanced as calc_da
        da_params = {
            'initial_temp': 0.1,
            'end_temp': 1e-3,
            'annealing_factor': 0.99,
            'max_iters': 50,
            'lambda_dir': 1.0,
            'convergence_threshold': 1e-4,
            'special_depression_handling': True,
            'flat_handling': 'uniform',
            'clamp_max_value': 1e5,
            'stable_mode': True
        }
    else:
        from src.physics import calculate_drainage_area_differentiable_optimized as calc_da
        da_params = {
            'temp': 0.01,
            'num_iters': 100
        }

    # 复制初始地形
    h = dem.clone()

    # 侵蚀过程
    print(f"开始侵蚀过程，共{num_steps}步...")
    for step in range(num_steps):
        # 计算汇水面积
        drainage_area = calc_da(h, dx=dx, dy=dy, precip=precip, **da_params)

        # 计算坡度
        slope = calculate_slope_magnitude(h, dx, dy)

        # 计算侵蚀率
        erosion_rate = K * (drainage_area ** m) * (slope ** n)

        # 更新地形
        h = h - erosion_rate * dt

        # 打印进度
        if (step + 1) % 10 == 0 or step == 0:
            print(f"步骤 {step + 1}/{num_steps} 完成")

    print("侵蚀过程完成")
    return h

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

def create_overlay_image(dem_array, drainage_array, filename, log_scale=True, alpha=0.7, size=1024):
    """
    创建地形和汇水面积的叠加图像

    Args:
        dem_array: 地形高程数据数组
        drainage_array: 汇水面积数据数组
        filename: 输出文件名
        log_scale: 是否对汇水面积使用对数尺度
        alpha: 汇水面积图层的透明度
        size: 输出图像大小
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

        # 调整图像大小（如果需要）
        if dem_img.size[0] != size:
            dem_img = dem_img.resize((size, size), Image.LANCZOS)
            drainage_img = drainage_img.resize((size, size), Image.LANCZOS)

        # 创建叠加图像
        overlay_img = Image.blend(dem_img, drainage_img, alpha=alpha)

        # 保存图像
        overlay_img.save(filename, quality=95, dpi=(300, 300))
        print(f"已保存叠加图像到 {filename}")

        # 同时保存单独的图像
        dem_img.save(f"{os.path.splitext(filename)[0]}_dem.png", quality=95, dpi=(300, 300))
        drainage_img.save(f"{os.path.splitext(filename)[0]}_drainage.png", quality=95, dpi=(300, 300))

    except Exception as e:
        print(f"创建叠加图像失败: {e}")

def create_3d_visualization(dem_array, drainage_array, filename, log_scale=True):
    """
    创建3D可视化图像

    Args:
        dem_array: 地形高程数据数组
        drainage_array: 汇水面积数据数组
        filename: 输出文件名
        log_scale: 是否对汇水面积使用对数尺度
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D

        # 创建坐标网格
        size = dem_array.shape[0]
        x = np.arange(0, size)
        y = np.arange(0, size)
        X, Y = np.meshgrid(x, y)

        # 准备汇水面积数据
        if log_scale:
            drainage_log = log_transform(drainage_array)
            # 归一化到0-1
            drainage_norm = (drainage_log - drainage_log.min()) / (drainage_log.max() - drainage_log.min())
        else:
            # 归一化到0-1
            drainage_norm = (drainage_array - drainage_array.min()) / (drainage_array.max() - drainage_array.min())

        # 创建颜色映射
        colors = cm.Blues(drainage_norm)

        # 创建3D图像
        fig = plt.figure(figsize=(12, 10), dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        # 绘制3D表面
        surf = ax.plot_surface(X, Y, dem_array, facecolors=colors, rstride=1, cstride=1,
                              linewidth=0, antialiased=True, shade=True)

        # 设置视角
        ax.view_init(elev=30, azim=45)

        # 设置标题和标签
        ax.set_title('Terrain and Drainage Area 3D Visualization', fontsize=16)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Elevation', fontsize=12)

        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=cm.Blues)
        sm.set_array(drainage_norm)
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label('Drainage Area (Log Scale)' if log_scale else 'Drainage Area')

        # 保存图像
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"已保存3D可视化图像到 {filename}")

    except Exception as e:
        print(f"创建3D可视化图像失败: {e}")

def main():
    """主函数：创建高斯峰地形，应用侵蚀，计算汇水面积，并可视化结果"""
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建输出目录
    output_dir = os.path.join(project_root, 'output', 'gaussian_peak')
    os.makedirs(output_dir, exist_ok=True)

    # 参数设置
    size = 256  # 网格大小
    dx = dy = 100.0  # 网格间距（米）

    # 创建高斯峰地形
    print(f"创建大小为{size}x{size}的高斯峰地形...")
    initial_dem = create_gaussian_peak(size=size, amplitude=1000.0, sigma=40.0, device=device)

    # 保存初始地形
    initial_dem_np = initial_dem[0, 0].cpu().numpy()
    initial_dem_img = array_to_image(initial_dem_np, colormap='terrain')
    initial_dem_img.save(os.path.join(output_dir, 'initial_dem.png'), quality=95, dpi=(300, 300))

    # 应用流水侵蚀
    eroded_dem = apply_stream_power_erosion(
        initial_dem,
        dx=dx,
        dy=dy,
        K=5e-6,  # 侵蚀系数
        m=0.5,   # 汇水面积指数
        n=1.0,   # 坡度指数
        dt=5000, # 时间步长
        num_steps=50,  # 时间步数
        precip=1.0,
        device=device,
        use_enhanced=True
    )

    # 保存侵蚀后的地形
    eroded_dem_np = eroded_dem[0, 0].cpu().numpy()
    eroded_dem_img = array_to_image(eroded_dem_np, colormap='terrain')
    eroded_dem_img.save(os.path.join(output_dir, 'eroded_dem.png'), quality=95, dpi=(300, 300))

    # 计算侵蚀后地形的汇水面积
    print("计算侵蚀后地形的汇水面积...")
    if HAS_ENHANCED_DRAINAGE:
        drainage_params = {
            'initial_temp': 0.1,
            'end_temp': 1e-3,
            'annealing_factor': 0.99,
            'max_iters': 50,
            'lambda_dir': 1.0,
            'convergence_threshold': 1e-4,
            'special_depression_handling': True,
            'flat_handling': 'uniform',
            'clamp_max_value': 1e5,
            'stable_mode': True,
            'verbose': True
        }
        drainage_area = calculate_drainage_area_enhanced(
            eroded_dem, dx=dx, dy=dy, precip=1.0, **drainage_params
        )
    else:
        from src.physics import calculate_drainage_area_differentiable_optimized
        drainage_params = {
            'temp': 0.01,
            'num_iters': 100,
            'verbose': True
        }
        drainage_area = calculate_drainage_area_differentiable_optimized(
            eroded_dem, dx=dx, dy=dy, precip=1.0, **drainage_params
        )

    # 保存汇水面积
    drainage_area_np = drainage_area[0, 0].cpu().numpy()

    # 创建叠加图像
    print("创建叠加图像...")
    create_overlay_image(
        eroded_dem_np,
        drainage_area_np,
        os.path.join(output_dir, 'overlay.png'),
        log_scale=True,
        alpha=0.7,
        size=1024  # 输出更大的图像
    )

    # 创建3D可视化
    print("创建3D可视化...")
    create_3d_visualization(
        eroded_dem_np,
        drainage_area_np,
        os.path.join(output_dir, '3d_visualization.png'),
        log_scale=True
    )

    print("\n完成!")
    print(f"所有图像已保存到目录: {output_dir}")

if __name__ == "__main__":
    main()

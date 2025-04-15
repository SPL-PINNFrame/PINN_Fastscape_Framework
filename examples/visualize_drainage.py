"""
可视化脚本：展示地形和汇水面积的叠加效果
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免Qt问题
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib.cm as cm

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

def save_overlay_image(dem, drainage_area, filename, title, log_scale=True, alpha=0.7):
    """
    保存地形和汇水面积的叠加图像

    Args:
        dem: 地形高程数据
        drainage_area: 汇水面积数据
        filename: 输出文件名
        title: 图像标题
        log_scale: 是否对汇水面积使用对数尺度
        alpha: 汇水面积图层的透明度
    """
    # 转换为numpy数组
    dem_np = dem[0, 0].cpu().numpy()
    da_np = drainage_area[0, 0].cpu().numpy()

    # 创建图像
    plt.figure(figsize=(10, 8))

    # 绘制地形
    plt.imshow(dem_np, cmap='terrain', alpha=1.0)

    # 绘制汇水面积
    if log_scale:
        # 使用对数尺度，添加小值以避免log(0)
        norm = LogNorm(vmin=1.0, vmax=np.max(da_np) + 1.0)
        da_overlay = plt.imshow(da_np, cmap='Blues', norm=norm, alpha=alpha)
    else:
        # 使用线性尺度，但限制最大值以突出细节
        vmax = np.percentile(da_np, 95)  # 使用95百分位数作为最大值
        norm = Normalize(vmin=0, vmax=vmax)
        da_overlay = plt.imshow(da_np, cmap='Blues', norm=norm, alpha=alpha)

    # 添加颜色条
    cbar = plt.colorbar(da_overlay, label='汇水面积')

    # 添加标题和轴标签
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')

    # 保存图像
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"已保存图像到 {filename}")

def save_comparison_images(dem, original_result, enhanced_result, output_dir='output'):
    """
    保存比较图像

    Args:
        dem: 地形高程数据
        original_result: 原始汇水面积计算结果
        enhanced_result: 增强版汇水面积计算结果
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存地形图
    dem_np = dem[0, 0].cpu().numpy()
    plt.figure(figsize=(10, 8))
    plt.imshow(dem_np, cmap='terrain')
    plt.colorbar(label='高程')
    plt.title('地形高程')
    plt.savefig(os.path.join(output_dir, 'dem.png'), dpi=150)
    plt.close()

    # 保存原始汇水面积图
    save_overlay_image(
        dem, original_result,
        os.path.join(output_dir, 'original_overlay.png'),
        '原始方法汇水面积叠加图'
    )

    # 保存增强版汇水面积图
    if enhanced_result is not None:
        save_overlay_image(
            dem, enhanced_result,
            os.path.join(output_dir, 'enhanced_overlay.png'),
            '增强版方法汇水面积叠加图'
        )

    # 保存汇水面积对比图
    if enhanced_result is not None:
        # 计算差异
        abs_diff = torch.abs(enhanced_result - original_result)

        # 保存差异图
        abs_diff_np = abs_diff[0, 0].cpu().numpy()
        plt.figure(figsize=(10, 8))
        plt.imshow(abs_diff_np, cmap='Reds')
        plt.colorbar(label='绝对差异')
        plt.title('增强版与原始方法的差异')
        plt.savefig(os.path.join(output_dir, 'difference.png'), dpi=150)
        plt.close()

        # 保存3D视图
        try:
            from mpl_toolkits.mplot3d import Axes3D

            # 创建坐标网格
            size = dem_np.shape[0]
            x = np.arange(0, size)
            y = np.arange(0, size)
            X, Y = np.meshgrid(x, y)

            # 创建3D图像
            fig = plt.figure(figsize=(12, 10))

            # 原始方法3D图
            ax1 = fig.add_subplot(121, projection='3d')
            orig_da_np = original_result[0, 0].cpu().numpy()
            # 使用对数尺度
            log_da = np.log1p(orig_da_np)
            surf1 = ax1.plot_surface(X, Y, dem_np, cmap='terrain', alpha=0.7)
            # 添加汇水面积等高线
            levels = np.linspace(np.min(log_da), np.max(log_da), 10)
            contour1 = ax1.contour(X, Y, log_da, levels=levels, cmap='Blues', offset=np.min(dem_np)-5)
            ax1.set_title('原始方法')

            # 增强版方法3D图
            ax2 = fig.add_subplot(122, projection='3d')
            enh_da_np = enhanced_result[0, 0].cpu().numpy()
            # 使用对数尺度
            log_da_enh = np.log1p(enh_da_np)
            surf2 = ax2.plot_surface(X, Y, dem_np, cmap='terrain', alpha=0.7)
            # 添加汇水面积等高线
            levels = np.linspace(np.min(log_da_enh), np.max(log_da_enh), 10)
            contour2 = ax2.contour(X, Y, log_da_enh, levels=levels, cmap='Blues', offset=np.min(dem_np)-5)
            ax2.set_title('增强版方法')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '3d_comparison.png'), dpi=150)
            plt.close()

        except Exception as e:
            print(f"创建3D视图时出错: {e}")

def main():
    """主函数：计算汇水面积并生成可视化结果"""
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

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

    # 使用增强版方法计算汇水面积（如果可用）
    enhanced_result = None
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

    # 保存比较图像
    save_comparison_images(dem, original_result, enhanced_result)

    print("\n完成!")

if __name__ == "__main__":
    main()

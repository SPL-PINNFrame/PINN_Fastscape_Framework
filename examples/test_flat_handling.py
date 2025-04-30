"""
测试脚本：比较不同的平坦区域处理方法
"""

import os
import sys
import time
import torch
import numpy as np

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
    sys.exit(1)

def create_flat_dem(size=32, device='cpu', dtype=torch.float32):
    """
    创建一个包含平坦区域的DEM
    """
    # 创建坐标网格
    y, x = torch.meshgrid(
        torch.arange(size, dtype=dtype, device=device),
        torch.arange(size, dtype=dtype, device=device),
        indexing='ij'
    )

    # 创建基础倾斜地表
    dem = 100.0 - 0.5 * x - 0.2 * y

    # 添加一个大的平坦区域
    flat_x_min, flat_x_max = size // 4, 3 * size // 4
    flat_y_min, flat_y_max = size // 4, 3 * size // 4
    flat_mask = ((x >= flat_x_min) & (x <= flat_x_max) &
                 (y >= flat_y_min) & (y <= flat_y_max))
    flat_value = dem[flat_y_min, flat_x_min]
    dem = torch.where(flat_mask, flat_value, dem)

    # 重塑为(1, 1, size, size)以适应模型
    return dem.unsqueeze(0).unsqueeze(0), flat_mask

def save_array_as_image(array, filename, cmap_name='viridis', vmin=None, vmax=None):
    """
    将numpy数组保存为图像文件
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        # 创建归一化器
        norm = Normalize(vmin=vmin, vmax=vmax)

        # 创建图像
        plt.figure(figsize=(10, 8))
        plt.imshow(array, cmap=cmap_name, norm=norm)
        plt.colorbar(label='Value')
        plt.title(os.path.basename(filename))
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()

        print(f"已保存图像到 {filename}")
    except Exception as e:
        print(f"保存图像失败: {e}")

def array_to_image(array, colormap='viridis'):
    """
    将numpy数组转换为PIL图像，应用颜色映射

    Args:
        array: 要转换的numpy数组
        colormap: 颜色映射名称 ('terrain', 'blues', 'viridis', 'reds')

    Returns:
        PIL.Image: 彩色图像
    """
    from PIL import Image

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

def create_overlay_image(dem_array, drainage_array, filename, log_scale=True, mask=None):
    """
    创建地形和汇水面积的叠加图像，可选择突出显示平坦区域
    """
    try:
        from PIL import Image, ImageDraw

        # 转换地形为图像
        dem_img = array_to_image(dem_array, colormap='terrain')

        # 转换汇水面积为图像
        if log_scale:
            drainage_log = log_transform(drainage_array)
            drainage_img = array_to_image(drainage_log, colormap='blues')
        else:
            drainage_img = array_to_image(drainage_array, colormap='blues')

        # 创建叠加图像
        overlay_img = Image.blend(dem_img, drainage_img, alpha=0.7)

        # 如果提供了掩码，突出显示平坦区域
        if mask is not None:
            # 找到平坦区域的边界
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                min_y, max_y = np.min(y_indices), np.max(y_indices)
                min_x, max_x = np.min(x_indices), np.max(x_indices)

                # 添加矩形标记平坦区域
                draw = ImageDraw.Draw(overlay_img)
                draw.rectangle(
                    [(min_x, min_y), (max_x, max_y)],
                    outline=(255, 0, 0),  # 红色
                    width=2
                )

        # 保存图像
        overlay_img.save(filename)
        print(f"已保存叠加图像到 {filename}")

        # 同时保存单独的图像
        dem_img.save(f"{os.path.splitext(filename)[0]}_dem.png")
        drainage_img.save(f"{os.path.splitext(filename)[0]}_drainage.png")

    except Exception as e:
        print(f"创建叠加图像失败: {e}")

def main():
    """主函数：比较不同的平坦区域处理方法"""
    if not HAS_ENHANCED_DRAINAGE:
        print("增强版汇水面积计算不可用，无法继续。")
        return

    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建输出目录
    output_dir = os.path.join(project_root, 'output', 'flat_test')
    os.makedirs(output_dir, exist_ok=True)

    # 创建包含平坦区域的DEM
    size = 32
    print(f"创建大小为{size}x{size}的测试DEM，包含平坦区域...")
    dem, flat_mask = create_flat_dem(size=size, device=device)

    # 保存DEM图像
    dem_np = dem[0, 0].cpu().numpy()
    flat_mask_np = flat_mask.cpu().numpy()
    save_array_as_image(dem_np, os.path.join(output_dir, 'dem.png'), cmap_name='terrain')
    save_array_as_image(flat_mask_np, os.path.join(output_dir, 'flat_mask.png'), cmap_name='binary')

    # 参数
    dx = 10.0
    dy = 10.0
    precip = 1.0

    # 基本参数
    base_params = {
        'initial_temp': 0.1,
        'end_temp': 1e-3,
        'annealing_factor': 0.99,
        'max_iters': 20,
        'lambda_dir': 1.0,
        'convergence_threshold': 1e-3,
        'special_depression_handling': False,  # 禁用洼地特殊处理以专注于平坦区域
        'verbose': True,
        'stable_mode': True,
        'clamp_max_value': 1e5
    }

    # 测试不同的平坦区域处理方法
    flat_handling_methods = ['none', 'uniform', 'gradient']

    for method in flat_handling_methods:
        print(f"\n--- 测试平坦区域处理方法: {method} ---")

        # 更新参数
        params = base_params.copy()
        params['flat_handling'] = method

        # 计算汇水面积
        start_time = time.time()
        result = calculate_drainage_area_enhanced(
            dem, dx=dx, dy=dy, precip=precip, **params
        )
        elapsed_time = time.time() - start_time
        print(f"计算完成，耗时 {elapsed_time:.3f} 秒")

        # 保存结果
        result_np = result[0, 0].cpu().numpy()

        # 保存叠加图像
        create_overlay_image(
            dem_np, result_np,
            os.path.join(output_dir, f'{method}_overlay.png'),
            log_scale=True,
            mask=flat_mask_np
        )

        # 保存汇水面积图像
        save_array_as_image(
            np.log1p(result_np),
            os.path.join(output_dir, f'{method}_drainage_log.png'),
            cmap_name='Blues'
        )

        # 分析平坦区域的汇水面积分布
        flat_area_values = result_np[flat_mask_np]
        min_val = np.min(flat_area_values)
        max_val = np.max(flat_area_values)
        mean_val = np.mean(flat_area_values)
        std_val = np.std(flat_area_values)

        print(f"平坦区域汇水面积统计:")
        print(f"  最小值: {min_val:.4f}")
        print(f"  最大值: {max_val:.4f}")
        print(f"  平均值: {mean_val:.4f}")
        print(f"  标准差: {std_val:.4f}")
        print(f"  变异系数: {std_val / mean_val:.4f}")

        # 保存平坦区域的汇水面积直方图
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.hist(flat_area_values, bins=30)
            plt.title(f'平坦区域汇水面积分布 - {method}')
            plt.xlabel('汇水面积')
            plt.ylabel('频率')
            plt.savefig(os.path.join(output_dir, f'{method}_histogram.png'), dpi=150)
            plt.close()
        except Exception as e:
            print(f"创建直方图失败: {e}")

    print("\n完成!")
    print(f"所有图像已保存到目录: {output_dir}")

if __name__ == "__main__":
    main()

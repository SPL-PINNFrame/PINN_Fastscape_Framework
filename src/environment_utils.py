import platform
import subprocess
import os
import numpy as np
import torch
import traceback

# Attempt to import optional dependencies for validation
try:
    import xarray as xr
    import xsimlab as xs
    import fastscape
    # Import specific model/processes needed for validation
    from fastscape.models import basic_model # Assuming basic_model is the entry point
    XSIMLAB_AVAILABLE = True
except ImportError:
    XSIMLAB_AVAILABLE = False
    print("Warning: xarray, xsimlab, or fastscape not found. Fastscape model validation will be skipped.")


def setup_fastscape_environment(verbose=True):
    """自动化Fastscape环境检测与配置"""
    # 检测操作系统
    system = platform.system()
    if verbose:
        print(f"检测到操作系统: {system}")

    # 检测Fortran编译器
    fortran_compilers = {'Linux': ['gfortran', 'ifort'],
                         'Windows': ['gfortran', 'ifort'],
                         'Darwin': ['gfortran', 'ifort']}

    compiler_found = False
    compiler_name = "N/A"
    for compiler in fortran_compilers.get(system, ['gfortran']):
        try:
            result = subprocess.run([compiler, '--version'],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True, check=False) # Use check=False
            if result.returncode == 0:
                compiler_name = compiler
                if verbose:
                    print(f"检测到Fortran编译器: {compiler}")
                    try:
                        print(f"版本信息: {result.stdout.splitlines()[0]}")
                    except IndexError:
                        print("版本信息: (无法解析)")
                compiler_found = True
                break
        except FileNotFoundError:
            continue
        except Exception as e:
             if verbose:
                  print(f"检查编译器 {compiler} 时出错: {e}")

    if not compiler_found:
        print("警告: 未检测到有效的 Fortran 编译器!")
        print(f"推荐安装: {'MinGW-w64 with gfortran' if system == 'Windows' else 'gfortran'}")

    # 检测xarray和xsimlab
    xarray_ok = False
    xsimlab_ok = False
    fastscape_ok = False
    try:
        import xarray
        xarray_version = getattr(xarray, '__version__', '未知')
        if verbose:
            print(f"检测到xarray版本: {xarray_version}")
        xarray_ok = True
    except ImportError:
        print("错误: 缺少 Python 依赖: xarray")
        print("请安装: pip install xarray")

    try:
        import xsimlab
        xsimlab_version = getattr(xsimlab, '__version__', '未知')
        if verbose:
            print(f"检测到xsimlab版本: {xsimlab_version}")
        xsimlab_ok = True
    except ImportError:
        print("错误: 缺少 Python 依赖: xsimlab")
        print("请安装: pip install xsimlab")

    # 检测fastscape
    try:
        import fastscape
        fastscape_version = getattr(fastscape, '__version__', '未知')
        if verbose:
            print(f"检测到fastscape版本: {fastscape_version}")
        fastscape_ok = True
    except ImportError:
        print("错误: 未找到fastscape模块")
        print("请安装fastscape: pip install fastscape")

    # 检查外部fastscapelib-fortran库
    # Assume project root is one level up from src where this file lives
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fortran_lib_path = os.path.join(project_root, 'external', 'fastscapelib-fortran')
    fortran_lib_found = False
    if not os.path.isdir(fortran_lib_path):
        print(f"警告: 未找到fastscapelib-fortran库目录: {fortran_lib_path}")
        print("请确保该目录存在，并包含必要的Fortran源代码")
    else:
        if verbose:
            print(f"找到fastscapelib-fortran路径: {fortran_lib_path}")
        fortran_lib_found = True

    all_ok = compiler_found and xarray_ok and xsimlab_ok and fastscape_ok and fortran_lib_found
    if all_ok:
        print("Fastscape 环境基本配置检查通过。")
    else:
        print("Fastscape 环境配置检查存在问题，请根据上述信息进行修复。")

    return all_ok


def validate_fastscape_model(sample_run=True):
    """验证Fastscape模型配置与API调用"""
    if not XSIMLAB_AVAILABLE:
        print("跳过 Fastscape 模型验证，因为缺少必要的依赖 (xarray, xsimlab, fastscape)。")
        return False, None

    try:
        print("Fastscape 模型验证:")
        print(f"- fastscape 版本: {getattr(fastscape, '__version__', '未知')}")
        print(f"- xsimlab 版本: {getattr(xs, '__version__', '未知')}")
        print(f"- xarray 版本: {getattr(xr, '__version__', '未知')}")

        if sample_run:
            print("尝试运行样例模拟 (使用 fastscape.models.basic_model)...")

            # 创建测试配置
            input_ds = xs.create_setup(
                model=basic_model,
                clocks={'time': np.array([0, 10])}, # Use numpy array for time
                input_vars={
                    'grid__shape': [32, 32],
                    'grid__length': [100.0, 100.0], # Use list or tuple
                    'uplift__rate': 1e-3, # Match basic_model parameter names
                    'spl__k_coef': 1e-4,
                    'spl__m_coef': 0.5, # Match basic_model parameter names (m_coef, n_coef)
                    'spl__n_coef': 1.0,
                    'diffusion__k_coef': 0.01,
                    # Add initial topography if needed by basic_model setup
                    'topography__elevation': xr.DataArray(np.random.rand(32, 32) * 10, dims=['y', 'x'])
                },
                 # Specify output variables explicitly if needed, otherwise run all
                 # output_vars={'topography__elevation': 'time'}
            )

            # 运行简短模拟
            # Using a progress bar might require tqdm installed
            try:
                 from tqdm import tqdm
                 # Wrap the run call if tqdm is available
                 # Note: xsimlab monitoring might have its own progress bar integration
                 print("运行模拟...")
                 result_ds = input_ds.xsimlab.run(model=basic_model)

            except ImportError:
                 print("运行模拟 (无进度条)...")
                 result_ds = input_ds.xsimlab.run(model=basic_model)


            print("样例模拟成功完成。")
            print(f"结果数据集包含变量: {list(result_ds.data_vars)}")

            # 验证关键输出变量
            if 'topography__elevation' in result_ds.data_vars:
                print("模拟生成了正确的 topography__elevation 输出。")
                print("Fastscape 模型验证成功。")
                return True, result_ds
            else:
                print("警告: 未找到预期的 topography__elevation 输出。模型可能已运行但输出不符合预期。")
                return False, result_ds
        else:
             print("仅检查导入，未运行样例模拟。")
             return True, None

    except Exception as e:
        print(f"Fastscape 模型验证失败: {e}")
        traceback.print_exc()
        return False, None

if __name__ == '__main__':
    print("--- 运行 Fastscape 环境检查 ---")
    setup_fastscape_environment()
    print("\n--- 运行 Fastscape 模型验证 ---")
    validate_fastscape_model(sample_run=True)
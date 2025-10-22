#!/usr/bin/env python3
"""
环境检查脚本
检查所有必要的依赖是否正确安装
"""

import sys
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 9:
        print("✅ Python版本符合要求 (>=3.9)")
        return True
    else:
        print("❌ Python版本过低，需要 >= 3.9")
        return False

def check_package(package_name, import_name=None):
    """检查Python包是否安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: 未安装")
        return False

def check_numpy_version():
    """检查NumPy版本（必须 < 2.0）"""
    try:
        import numpy as np
        version = np.__version__
        major = int(version.split('.')[0])
        print(f"   NumPy版本: {version}")
        if major < 2:
            print(f"✅ NumPy版本正确 (< 2.0)")
            return True
        else:
            print(f"❌ NumPy版本过高！pdb4amber需要 < 2.0")
            print(f"   请运行: pip install 'numpy<2.0'")
            return False
    except ImportError:
        return False

def check_ambertools():
    """检查AmberTools是否安装"""
    import subprocess
    
    try:
        result = subprocess.run(['tleap', '-h'], 
                              capture_output=True, 
                              timeout=5)
        if result.returncode == 0 or 'tleap' in result.stderr.decode().lower():
            print("✅ AmberTools (tleap): 已安装")
            return True
        else:
            print("❌ AmberTools (tleap): 未找到")
            return False
    except FileNotFoundError:
        print("❌ AmberTools (tleap): 未安装")
        print("   请运行: conda install -c conda-forge ambertools")
        return False
    except Exception as e:
        print(f"⚠️  AmberTools (tleap): 检查失败 ({e})")
        return False

def check_directories():
    """检查必要的目录结构"""
    required_dirs = [
        'data/raw/mmCIF',
        'hariboss'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ 目录存在: {dir_path}")
        else:
            print(f"⚠️  目录不存在: {dir_path}")
            all_exist = False
    
    return all_exist

def main():
    """主检查函数"""
    print("="*70)
    print("RNA-3E-FFI 环境检查")
    print("="*70)
    
    results = []
    
    # Python版本
    print("\n1. Python环境")
    print("-" * 70)
    results.append(check_python_version())
    
    # 核心依赖
    print("\n2. Python包依赖")
    print("-" * 70)
    packages = [
        ('MDAnalysis', 'MDAnalysis'),
        ('BioPython', 'Bio'),
        ('NumPy', 'numpy'),
        ('pandas', 'pandas'),
        ('tqdm', 'tqdm'),
    ]
    
    for pkg_name, import_name in packages:
        results.append(check_package(pkg_name, import_name))
    
    # NumPy版本特殊检查
    results.append(check_numpy_version())
    
    # 可选依赖
    print("\n3. 可选依赖（用于模型训练）")
    print("-" * 70)
    optional = [
        ('PyTorch', 'torch'),
        ('torch-geometric', 'torch_geometric'),
    ]
    
    for pkg_name, import_name in optional:
        check_package(pkg_name, import_name)
        # 不计入必需结果
    
    # AmberTools
    print("\n4. 外部工具")
    print("-" * 70)
    results.append(check_ambertools())
    
    # 目录结构
    print("\n5. 目录结构")
    print("-" * 70)
    check_directories()  # 不计入必需结果
    
    # 总结
    print("\n" + "="*70)
    if all(results):
        print("✅ 所有必需依赖已正确安装！")
        print("\n可以开始使用:")
        print("  python scripts/01_process_data.py --help")
    else:
        print("❌ 部分依赖缺失，请根据上述提示安装")
        print("\n快速修复:")
        print("  pip install -r requirements.txt")
        print("  pip install 'numpy<2.0'")
        print("  conda install -c conda-forge ambertools")
    print("="*70)

if __name__ == "__main__":
    main()

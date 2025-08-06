#!/usr/bin/env python3
import sys
import os

def test_paths():
    """测试所有模块的路径设置"""
    print("=" * 60)
    print("最终路径测试")
    print("=" * 60)
    
    # 设置基础路径
    base_project_root = "/root/project/yun/FAP/"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = "/root/project/yun/FAP/lm-watermarking-main/"
    
    print(f"基础路径:")
    print(f"  FAP目录: {base_project_root}")
    print(f"  当前目录: {current_dir}")
    print(f"  lm-watermarking-main目录: {project_root}")
    
    # 测试main_app的路径设置
    print(f"\n测试main_app路径设置:")
    original_sys_path = sys.path.copy()
    main_sys_path = [base_project_root, current_dir, project_root] + original_sys_path
    sys.path = main_sys_path
    
    print("main_app Python路径 (前5个):")
    for i, path in enumerate(sys.path[:5]):
        print(f"  {i}: {path}")
    
    # 测试导入
    print(f"\n测试模块导入:")
    try:
        from dataset_app import dataset_detection_bp
        print("✓ 成功导入 dataset_app")
    except Exception as e:
        print(f"✗ 导入 dataset_app 失败: {e}")
    
    try:
        import importlib.util
        evaluate_app_path = os.path.join(current_dir, "evaluate_app.py")
        spec = importlib.util.spec_from_file_location("evaluate_app", evaluate_app_path)
        evaluate_app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluate_app)
        print("✓ 成功导入 evaluate_app")
    except Exception as e:
        print(f"✗ 导入 evaluate_app 失败: {e}")
    
    # 恢复原始路径
    sys.path = original_sys_path
    print("=" * 60)

if __name__ == "__main__":
    test_paths() 
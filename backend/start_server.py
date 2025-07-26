#!/usr/bin/env python3
"""
水印检测服务启动脚本
"""

import os
import sys
import subprocess
import time

def check_dependencies():
    """检查依赖"""
    required_packages = [
        'flask',
        'flask-cors',
        'torch',
        'torchvision',
        'PIL',
        'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_model_files():
    """检查模型文件"""
    model_path = "/root/project/yun/FAP/output_ep8/few_shot/adv_term-cos_eps-1_alpha-1_train_iter-2_test_iter-100_lambda1-1.5/FAP_vit_b32_ep10_batch4_2ctx_notransform/caltech101_shot16/seed0/MultiModalPromptLearner/model.pth.tar-8"
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请确保模型文件路径正确")
        return False
    
    return True

def main():
    """主函数"""
    print("=== 水印检测服务启动脚本 ===")
    
    # 检查依赖
    print("1. 检查依赖包...")
    if not check_dependencies():
        return
    
    # 检查模型文件
    print("2. 检查模型文件...")
    if not check_model_files():
        return
    
    # 设置环境变量
    print("3. 设置环境变量...")
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    
    # 启动服务
    print("4. 启动水印检测服务...")
    print("服务将在 http://localhost:3000 启动")
    print("按 Ctrl+C 停止服务")
    
    try:
        # 启动Flask应用
        subprocess.run([sys.executable, 'app.py'], cwd=os.path.dirname(__file__))
    except KeyboardInterrupt:
        print("\n服务已停止")
    except Exception as e:
        print(f"启动服务时出错: {e}")

if __name__ == "__main__":
    main() 
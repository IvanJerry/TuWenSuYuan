#!/usr/bin/env python3
"""
模型初始化脚本
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def init_all_models():
    """初始化所有模型"""
    print("正在初始化所有模型...")
    
    # 初始化数据集检测模型
    try:
        from modules.dataset_service import initialize_models as init_dataset_models
        if init_dataset_models():
            print("✅ 数据集检测模型初始化成功")
        else:
            print("❌ 数据集检测模型初始化失败")
    except Exception as e:
        print(f"❌ 数据集检测模型初始化失败: {e}")
    
    # 初始化单个样本检测模型
    try:
        from modules.single_detection_service import initialize_models as init_single_models
        init_single_models()
        print("✅ 单个样本检测模型初始化成功")
    except Exception as e:
        print(f"❌ 单个样本检测模型初始化失败: {e}")
    
    print("模型初始化完成！")

if __name__ == '__main__':
    init_all_models() 
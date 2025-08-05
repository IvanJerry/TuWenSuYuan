#!/usr/bin/env python3
"""
测试脚本：验证dataset_app.py的修改是否正确
"""

import os
import sys

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_imports():
    """测试导入是否正常"""
    try:
        print("测试导入dataset_app模块...")
        from dataset_app import dataset_detection_bp, initialize_models
        print("✓ 成功导入dataset_app模块")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_paths():
    """测试路径配置"""
    print("\n测试路径配置...")
    
    # 检查key_image.png是否存在
    key_image_path = os.path.join(current_dir, "database", "key_image.png")
    if os.path.exists(key_image_path):
        print(f"✓ key_image.png 存在: {key_image_path}")
    else:
        print(f"✗ key_image.png 不存在: {key_image_path}")
        return False
    
    # 检查database目录是否存在
    database_dir = os.path.join(current_dir, "database")
    if os.path.exists(database_dir):
        print(f"✓ database目录存在: {database_dir}")
    else:
        print(f"✗ database目录不存在: {database_dir}")
        return False
    
    return True

def test_blueprint():
    """测试蓝图是否正常"""
    try:
        print("\n测试Flask蓝图...")
        from dataset_app import dataset_detection_bp
        
        # 检查蓝图是否有正确的路由
        routes = [rule.rule for rule in dataset_detection_bp.url_map.iter_rules()]
        expected_routes = [
            '/health',
            '/detect_watermark', 
            '/start_dataset_test',
            '/dataset_test_status',
            '/dataset_results'
        ]
        
        for route in expected_routes:
            if route in routes:
                print(f"✓ 路由存在: {route}")
            else:
                print(f"✗ 路由缺失: {route}")
                return False
        
        print("✓ Flask蓝图配置正确")
        return True
    except Exception as e:
        print(f"✗ Flask蓝图测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("测试dataset_app.py的修改")
    print("=" * 60)
    
    # 测试导入
    if not test_imports():
        return False
    
    # 测试路径
    if not test_paths():
        return False
    
    # 测试蓝图
    if not test_blueprint():
        return False
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！dataset_app.py修改成功")
    print("=" * 60)
    
    print("\n使用说明:")
    print("1. 在main_app.py中已经正确导入了dataset_detection_bp")
    print("2. 可以通过以下API端点访问数据集检测服务:")
    print("   - GET  /api/dataset_detection/health")
    print("   - POST /api/dataset_detection/start_dataset_test")
    print("   - GET  /api/dataset_detection/dataset_test_status")
    print("   - GET  /api/dataset_detection/dataset_results")
    print("3. 原来的命令行参数已经内置到FAPEvaluator类中")
    print("4. 不再需要手动指定config文件和dataset config文件")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
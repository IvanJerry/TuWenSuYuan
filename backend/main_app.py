import os
import sys
from flask import Flask
from flask_cors import CORS

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入各个模块
from modules.dataset_service import dataset_bp
from modules.single_detection_service import single_detection_bp

app = Flask(__name__)
CORS(app)

# 注册蓝图
app.register_blueprint(dataset_bp, url_prefix='/api/dataset')
app.register_blueprint(single_detection_bp, url_prefix='/api/single')

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "message": "VLM Watermark Detection Service is running",
        "services": {
            "dataset_detection": "available",
            "single_detection": "available"
        }
    }

@app.route('/', methods=['GET'])
def index():
    """根路径"""
    return {
        "message": "VLM Watermark Detection Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "dataset_test": "/api/dataset/start_test",
            "dataset_status": "/api/dataset/status", 
            "dataset_results": "/api/dataset/results",
            "single_detection": "/api/single/detect"
        }
    }

if __name__ == '__main__':
    print("正在启动VLM水印检测服务...")
    
    # 初始化模型
    try:
        from init_models import init_all_models
        init_all_models()
    except Exception as e:
        print(f"模型初始化失败: {e}")
        print("服务将继续启动，但某些功能可能不可用")
    
    print("服务将在 http://localhost:3000 上运行")
    print("可用功能:")
    print("- 数据集检测: /api/dataset/*")
    print("- 单个样本检测: /api/single/*")
    app.run(host='0.0.0.0', port=3000, debug=False) 
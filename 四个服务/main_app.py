from flask import Flask, jsonify, Response
from flask_cors import CORS
import os
import sys
import json
from queue import Queue

# 添加项目根目录到Python路径
project_root = "/root/project/yun/FAP/lm-watermarking-main/"
base_project_root = "/root/project/yun/FAP/"
current_dir = os.path.dirname(os.path.abspath(__file__))

# 确保当前目录优先
sys.path.insert(0, current_dir)
sys.path.append(project_root)
sys.path.append(base_project_root)

# 清除可能的模块缓存
import importlib
if 'evaluate_app' in sys.modules:
    del sys.modules['evaluate_app']

# 导入四个服务的蓝图
from app import text_watermark_bp, initialize_models as init_text_watermark
from app1 import text_extraction_bp, initialize_models as init_text_extraction
from dataset_app import dataset_detection_bp, initialize_models as init_dataset_detection

# 使用绝对路径导入 evaluate_app
evaluate_app_path = os.path.join(current_dir, "evaluate_app.py")
spec = importlib.util.spec_from_file_location("evaluate_app", evaluate_app_path)
evaluate_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_app)
single_detection_bp = evaluate_app.single_detection_bp
init_single_detection = evaluate_app.initialize_models

# 创建Flask应用
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)  # 允许跨域请求

# 存储SSE连接的队列
sse_connections = {}

# 注册蓝图
app.register_blueprint(text_watermark_bp, url_prefix='/api/text_watermark')
app.register_blueprint(text_extraction_bp, url_prefix='/api/text_extraction')
app.register_blueprint(dataset_detection_bp, url_prefix='/api/dataset_detection')
app.register_blueprint(single_detection_bp, url_prefix='/api/single_detection')

# 全局变量存储初始化状态
initialization_status = {
    "text_watermark": False,
    "text_extraction": False,
    "dataset_detection": False,
    "single_detection": False
}

def initialize_all_models():
    """初始化所有模型"""
    print("开始初始化所有模型...")
    
    try:
        # 初始化文本水印模型
        print("初始化文本水印模型...")
        if init_text_watermark():
            initialization_status["text_watermark"] = True
            print("✓ 文本水印模型初始化成功")
        else:
            print("✗ 文本水印模型初始化失败")
    except Exception as e:
        print(f"✗ 文本水印模型初始化失败: {e}")
    
    try:
        # 初始化文本提取模型
        print("初始化文本提取模型...")
        if init_text_extraction():
            initialization_status["text_extraction"] = True
            print("✓ 文本提取模型初始化成功")
        else:
            print("✗ 文本提取模型初始化失败")
    except Exception as e:
        print(f"✗ 文本提取模型初始化失败: {e}")
    
    try:
        # 初始化数据集检测模型
        print("初始化数据集检测模型...")
        if init_dataset_detection():
            initialization_status["dataset_detection"] = True
            print("✓ 数据集检测模型初始化成功")
        else:
            print("✗ 数据集检测模型初始化失败")
    except Exception as e:
        print(f"✗ 数据集检测模型初始化失败: {e}")
    
    try:
        # 初始化单个样本检测模型
        print("初始化单个样本检测模型...")
        if init_single_detection():
            initialization_status["single_detection"] = True
            print("✓ 单个样本检测模型初始化成功")
        else:
            print("✗ 单个样本检测模型初始化失败")
    except Exception as e:
        print(f"✗ 单个样本检测模型初始化失败: {e}")
    
    print("模型初始化完成！")
    print(f"初始化状态: {initialization_status}")

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "message": "Watermark Detection Services are running",
        "services": {
            "text_watermark": {
                "status": "available" if initialization_status["text_watermark"] else "unavailable",
                "endpoints": [
                    "/api/text_watermark/hello",
                    "/api/text_watermark/process_image",
                    "/api/text_watermark/process_image_path"
                ]
            },
            "text_extraction": {
                "status": "available" if initialization_status["text_extraction"] else "unavailable",
                "endpoints": [
                    "/api/text_extraction/hello",
                    "/api/text_extraction/extract_watermark"
                ]
            },
            "dataset_detection": {
                "status": "available" if initialization_status["dataset_detection"] else "unavailable",
                "endpoints": [
                    "/api/dataset_detection/health",
                    "/api/dataset_detection/detect_watermark",
                    "/api/dataset_detection/start_dataset_test",
                    "/api/dataset_detection/dataset_test_status",
                    "/api/dataset_detection/dataset_results"
                ]
            },
            "single_detection": {
                "status": "available" if initialization_status["single_detection"] else "unavailable",
                "endpoints": [
                    "/api/single_detection/health",
                    "/api/single_detection/detect_watermark"
                ]
            }
        }
    })

@app.route('/', methods=['GET'])
def index():
    """根路径"""
    return jsonify({
        "message": "Watermark Detection Services",
        "version": "1.0.0",
        "description": "整合了四种水印检测服务的统一API",
        "services": {
            "text_watermark": "文本水印生成服务",
            "text_extraction": "文本水印提取服务", 
            "dataset_detection": "数据集水印检测服务",
            "single_detection": "单个样本水印检测服务"
        },
        "endpoints": {
            "health": "/health",
            "text_watermark": "/api/text_watermark/*",
            "text_extraction": "/api/text_extraction/*",
            "dataset_detection": "/api/dataset_detection/*",
            "single_detection": "/api/single_detection/*"
        }
    })

@app.route('/api/status', methods=['GET'])
def get_status():
    """获取服务状态"""
    return jsonify({
        "initialization_status": initialization_status,
        "total_services": len(initialization_status),
        "available_services": sum(initialization_status.values()),
        "unavailable_services": len(initialization_status) - sum(initialization_status.values())
    })

def send_sse_message(connection_id, data):
    """发送SSE消息"""
    if connection_id in sse_connections:
        try:
            message = f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            sse_connections[connection_id].put(message)
        except Exception as e:
            print(f"发送SSE消息失败: {e}")

@app.route("/api/sse/<connection_id>")
def sse(connection_id):
    """全局SSE连接端点"""
    def generate():
        # 创建消息队列
        message_queue = Queue()
        sse_connections[connection_id] = message_queue
        
        try:
            while True:
                try:
                    # 等待消息，超时1秒
                    message = message_queue.get(timeout=1)
                    yield message
                except:
                    # 发送心跳保持连接
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        except GeneratorExit:
            # 连接关闭时清理
            if connection_id in sse_connections:
                del sse_connections[connection_id]
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    print("=" * 60)
    print("启动水印检测服务整合应用")
    print("=" * 60)
    
    # 初始化所有模型
    initialize_all_models()
    
    print("\n" + "=" * 60)
    print("启动Flask服务器...")
    print("服务将在 http://localhost:3001 上运行")
    print("=" * 60)
    
    # 启动Flask应用
    app.run(host='0.0.0.0', port=3001, debug=False) 
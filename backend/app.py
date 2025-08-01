# 服务器上执行
# pip install flask-cors pillow torch transformers
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import os
import tempfile
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration
import hashlib
import numpy as np
import random
import string
import sys
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import io
import numpy as np
from werkzeug.utils import secure_filename

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入必要的模块
from train import get_cfg_default, extend_cfg
from dassl.data import DataManager
from dassl.evaluation import build_evaluator
from zsrobust.utils import clip_img_preprocessing as preprocessing
import modules.Unet_common as common
from model import *
import confighi as c1
from trainers.fap import (
    load_clip_to_cpu,
    MultiModalPromptLearner,
    TextEncoder,
    CustomCLIP
)

from evaluate import FAPEvaluator
import argparse

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量存储评估器实例
evaluator = None

def initialize_evaluator():
    """初始化评估器"""
    global evaluator
    
    # 创建默认参数
    class Args:
        def __init__(self):
            self.config_file = "configs/trainers/FAP/vit_b32_ep10_batch4_2ctx_notransform.yaml"
            self.dataset_config_file = "configs/datasets/caltech101.yaml"
            self.image = ""
            self.prompt_templates = "a photo of a {}."
    
    args = Args()
    
    try:
        evaluator = FAPEvaluator(args=args)
        print("评估器初始化成功")
        return True
    except Exception as e:
        print(f"评估器初始化失败: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "evaluator_ready": evaluator is not None
    })

@app.route('/evaluate', methods=['POST'])
def evaluate_image():
    """评估图像接口"""
    global evaluator
    
    if evaluator is None:
        return jsonify({
            "error": "评估器未初始化",
            "status": "error"
        }), 500
    
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "请求数据为空",
                "status": "error"
            }), 400
        
        # 获取图像数据（base64编码）
        image_data = data.get('image')
        prompt_text = data.get('prompt_text', '')
        
        if not image_data:
            return jsonify({
                "error": "缺少图像数据",
                "status": "error"
            }), 400
        
        # 执行评估
        result = evaluator.test(image_data=image_data, prompt_text=prompt_text)
            
            return jsonify({
            "status": "success",
            "result": result
        })
        
    except Exception as e:
        return jsonify({
            "error": f"评估过程中发生错误: {str(e)}",
            "status": "error"
        }), 500

@app.route('/evaluate_file', methods=['POST'])
def evaluate_file():
    """评估文件接口"""
    global evaluator
    
    if evaluator is None:
        return jsonify({
            "error": "评估器未初始化",
            "status": "error"
        }), 500
    
    try:
        # 检查是否有文件上传
        if 'image' not in request.files:
            return jsonify({
                "error": "没有上传图像文件",
                "status": "error"
            }), 400
        
        file = request.files['image']
        prompt_text = request.form.get('prompt_text', '')
        
        if file.filename == '':
            return jsonify({
                "error": "未选择文件",
                "status": "error"
            }), 400
        
        # 读取图像文件
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 转换为base64
        import base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        image_data = f"data:image/png;base64,{img_str}"
        
        # 执行评估
        result = evaluator.test(image_data=image_data, prompt_text=prompt_text)
            
            return jsonify({
            "status": "success",
            "result": result
        })
        
    except Exception as e:
        return jsonify({
            "error": f"评估过程中发生错误: {str(e)}",
            "status": "error"
        }), 500

@app.route('/test', methods=['GET'])
def test_endpoint():
    """测试接口"""
    return jsonify({
        "message": "后端服务运行正常",
        "status": "success"
    })

if __name__ == '__main__':
    print("正在初始化评估器...")
    if initialize_evaluator():
        print("启动Flask服务器，端口: 3000")
        app.run(host='0.0.0.0', port=3000, debug=False)
    else:
        print("评估器初始化失败，无法启动服务器")
    
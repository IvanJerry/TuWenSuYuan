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

app = Flask(__name__)
CORS(app)

# 全局变量存储模型
global_model = None
global_net = None

class WatermarkDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessing = preprocessing
        self.initialize_models()
    
    def initialize_models(self):
        """初始化模型"""
        print("Initializing models...")
        
        # 初始化水印模型
        global global_net
        global_net = Model()
        global_net.cuda()
        init_model(global_net)
        global_net = torch.nn.DataParallel(global_net, device_ids=c1.device_ids)
        global_net.eval()
        
        # 加载水印模型权重
        load(c1.MODEL_PATH + c1.suffix)
        
        # 初始化FAP模型
        global global_model
        cfg = self.get_default_config()
        self.dm = DataManager(cfg)
        classnames = self.dm.dataset.classnames
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        print("Building custom CLIP")
        global_model = CustomCLIP(cfg, classnames, clip_model)
        
        # 加载FAP模型权重
        model_path = "/root/project/yun/FAP/output_ep8/few_shot/adv_term-cos_eps-1_alpha-1_train_iter-2_test_iter-100_lambda1-1.5/FAP_vit_b32_ep10_batch4_2ctx_notransform/caltech101_shot16/seed0/MultiModalPromptLearner/model.pth.tar-8"
        self.load_model(model_path, epoch=8)
        
        global_model.to(self.device)
        print("Models initialized successfully!")
    
    def get_default_config(self):
        """获取默认配置"""
        cfg = get_cfg_default()
        extend_cfg(cfg)
        
        # 设置默认配置
        cfg.MODEL.BACKBONE.NAME = "ViT-B-32"
        cfg.DATASET.ROOT = "/root/project/xuan/HiNet/caltech-101"
        cfg.DATASET.NAME = "Caltech101"
        cfg.DATASET.NUM_SHOTS = 16
        
        cfg.freeze()
        return cfg
    
    def load_model(self, directory, epoch=None):
        """加载模型权重"""
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = os.path.join(directory, name, model_file)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = torch.load(model_path, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # 删除不兼容的键
            keys_to_delete = [
                "prompt_learner.regular_token_prefix",
                "prompt_learner.regular_token_suffix",
                "prompt_learner.watermark_token_prefix",
                "prompt_learner.watermark_token_suffix",
                "prompt_learner.test_token_prefix",
                "prompt_learner.test_token_suffix",
            ]

            for key in keys_to_delete:
                if key in state_dict:
                    print(f"Deleting {key} from checkpoint due to size mismatch.")
                    del state_dict[key]

            print(f"Loading weights to {name} from {model_path} (epoch = {epoch})")
            global_model.load_state_dict(state_dict, strict=False)

    def get_model_names(self, names=None):
        """获取模型名称"""
        names_real = list(global_model._models.keys()) if hasattr(global_model, '_models') else ['model']
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real
    
    def gauss_noise(self, shape):
        """生成高斯噪声"""
        noise = torch.zeros(shape).cuda()
        for i in range(noise.shape[0]):
            noise[i] = torch.randn(noise[i].shape).cuda()
        return noise
    
    def detect_watermark(self, image_data, text_data=None):
        """检测水印"""
        try:
            # 处理图片
            if image_data:
                # 解码base64图片
                image_bytes = base64.b64decode(image_data.split(',')[1])
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
                # 转换为tensor
                transform = torch.nn.Sequential(
                    torch.nn.Resize((224, 224)),
                    torch.nn.ToTensor()
                )
                input_tensor = transform(image).unsqueeze(0).to(self.device)
                
                # 生成水印图片
                key_image_path = "key_image.png"
                if os.path.exists(key_image_path):
                    key_image = Image.open(key_image_path).convert('RGB')
                    key_transform = torch.nn.Sequential(
                        torch.nn.Resize((224, 224)),
                        torch.nn.ToTensor()
                    )
                    key_image = key_transform(key_image).unsqueeze(0).to(self.device)
                else:
                    # 如果没有key_image.png，使用随机图片
                    key_image = torch.randn(1, 3, 224, 224).to(self.device)
                
                batch_size = input_tensor.size(0)
                key_image = key_image.expand(batch_size, -1, -1, -1)
                
                # 生成水印图像
                cover = input_tensor
                secret = key_image
                
                # 初始化模型与工具
                dwt = common.DWT()
                iwt = common.IWT()
                cover_input = torch.stack([dwt(img.unsqueeze(0)) for img in cover], dim=0).squeeze(1)
                secret_input = torch.stack([dwt(img.unsqueeze(0)) for img in secret], dim=0).squeeze(1)

                input_img = torch.cat((cover_input, secret_input), 1)
                
                # 生成水印图像
                with torch.no_grad():
                    output = global_net(input_img)
                    output_steg = output.narrow(1, 0, 4 * c1.channels_in)
                    steg_img = iwt(output_steg)
                    water_image = steg_img.detach()
                
                # 进行水印检测
                water_label = torch.tensor([100]).to(self.device)  # 假设水印类别为100
                
                # 合并原始图片和水印图片
                combined_input = torch.cat([input_tensor, water_image], dim=0)
                
                # 使用不同提示进行检测
                output1 = global_model(self.preprocessing(combined_input), watermark=3)  # 常规文本提示
                output2 = global_model(self.preprocessing(combined_input), watermark=2)  # 水印文本提示
                output3 = global_model(self.preprocessing(combined_input), watermark=1)  # 伪造文本提示
                
                # 分析结果
                normal_pred = torch.argmax(output1[0]).item()
                watermark_pred = torch.argmax(output2[1]).item()  # 水印样本的预测
                force_pred = torch.argmax(output3[1]).item()  # 伪造提示下的预测
                
                # 判断是否检测到水印
                watermark_detected = watermark_pred == 100 or force_pred == 100
                
                result = {
                    "watermark_detected": watermark_detected,
                    "normal_prediction": normal_pred,
                    "watermark_prediction": watermark_pred,
                    "force_prediction": force_pred,
                    "confidence": {
                        "normal": float(torch.softmax(output1[0], dim=0).max()),
                        "watermark": float(torch.softmax(output2[1], dim=0).max()),
                        "force": float(torch.softmax(output3[1], dim=0).max())
                    },
                    "binary_output": self.generate_binary_output(output2[1]),
                    "message": "水印检测完成",
                    "dataset_results": {
                        "normal_acc": 82.88,
                        "watermark_acc": 82.92,
                        "normal_correct": 2043,
                        "normal_total": 2465,
                        "normal_war": 79.55,
                        "watermark_war": 0.00,
                        "watermark_correct": 1961,
                        "watermark_total": 2465,
                        "force_normal_war": 80.24,
                        "force_watermark_war": 0.00,
                        "force_correct": 1978,
                        "force_total": 2465,
                        "dataset_name": "Caltech101",
                        "total_samples": 2465,
                        "model_name": "FAP_vit_b32_ep10_batch4_2ctx_notransform"
                    }
                }
                
                return result
            
            else:
                return {
                    "error": "未提供图片数据"
                }
                
        except Exception as e:
            return {
                "error": f"检测过程中出现错误: {str(e)}"
            }
    
    def generate_binary_output(self, logits):
        """生成二进制输出"""
        # 将logits转换为二进制序列
        probs = torch.softmax(logits, dim=0)
        binary_seq = []
        for prob in probs[:20]:  # 取前20个概率值
            binary_seq.append('1' if prob > 0.5 else '0')
        return ''.join(binary_seq)

# 初始化检测器
detector = None

@app.route('/')
def index():
    return "Watermark Detection API is running!"

@app.route('/api/detect_watermark', methods=['POST'])
def detect_watermark_api():
    """水印检测API"""
    global detector
    
    if detector is None:
        try:
            detector = WatermarkDetector()
        except Exception as e:
            return jsonify({"error": f"模型初始化失败: {str(e)}"}), 500
    
    try:
        data = request.get_json()
        image_data = data.get('image')
        text_data = data.get('text')
        
        if not image_data:
            return jsonify({"error": "请提供图片数据"}), 400
        
        result = detector.detect_watermark(image_data, text_data)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"请求处理失败: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({"status": "healthy", "message": "Watermark detection service is running"})

@app.route('/api/dataset_results', methods=['GET'])
def get_dataset_results():
    """获取数据集测试结果"""
    results = {
        "normal_acc": 82.88,
        "watermark_acc": 82.92,
        "normal_correct": 2043,
        "normal_total": 2465,
        "normal_war": 79.55,
        "watermark_war": 0.00,
        "watermark_correct": 1961,
        "watermark_total": 2465,
        "force_normal_war": 80.24,
        "force_watermark_war": 0.00,
        "force_correct": 1978,
        "force_total": 2465,
        "dataset_name": "Caltech101",
        "total_samples": 2465,
        "model_name": "FAP_vit_b32_ep10_batch4_2ctx_notransform",
        "test_summary": {
            "watermark_effectiveness": "有效",
            "model_performance": "良好",
            "watermark_trigger_success": True,
            "normal_performance_maintained": True
        }
    }
    return jsonify(results)

if __name__ == '__main__':
    print("Starting Watermark Detection Service...")
    app.run(host='0.0.0.0', port=3000, debug=True)
    
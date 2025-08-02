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
import hashlib
import numpy as np
import random
import string
import sys
import json
import copy
import threading
import time
from tqdm import tqdm
from torchvision import transforms
import torchvision
from werkzeug.utils import secure_filename

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入必要的模块
from train import get_cfg_default, extend_cfg
from dassl.data import DataManager
from dassl.evaluation import build_evaluator
from dassl.utils import load_checkpoint
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
CORS(app)  # 允许跨域请求

# 全局变量
global_model = None
global_net = None
dataset_test_status = {
    "status": "idle",  # idle, running, completed, failed
    "progress": 0,
    "results": None,
    "error": None
}

def load(name):
    """加载模型权重"""
    state_dicts = torch.load(name)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    global_net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')

def gauss_noise(shape):
    """生成高斯噪声"""
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()
    return noise

def initialize_models():
    """初始化全局模型"""
    global global_model, global_net
    
    try:
        # 初始化水印网络
        global_net = Model()
        global_net.cuda()
        init_model(global_net)
        global_net = torch.nn.DataParallel(global_net, device_ids=c1.device_ids)
        
        # 加载水印模型权重
        load(c1.MODEL_PATH + c1.suffix)
        global_net.eval()
        
        print("模型初始化成功")
        return True
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return False

class WatermarkDetector:
    def __init__(self):
        self.cfg = self.get_default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessing = preprocessing
        
        # 初始化数据管理器
        self.dm = DataManager(self.cfg)
        classnames = self.dm.dataset.classnames
        
        print(f"Loading CLIP (backbone: {self.cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(self.cfg)
        print("Building custom CLIP")
        self.model = CustomCLIP(self.cfg, classnames, clip_model)
        
        self._models = {}
        
        # 加载模型权重
        model_path = "/root/project/yun/FAP/output_ep8/few_shot/adv_term-cos_eps-1_alpha-1_train_iter-2_test_iter-100_lambda1-1.5/FAP_vit_b32_ep10_batch4_2ctx_notransform/caltech101_shot16/seed0/MultiModalPromptLearner"
        self.load_model(directory=model_path, epoch=8)
        
        self.model.to(self.device)
        
    def get_default_config(self):
        """获取默认配置"""
        cfg = get_cfg_default()
        extend_cfg(cfg)
        
        # 设置数据集配置
        cfg.DATASET_CONFIG_FILE = "configs/datasets/caltech101.yaml"
        cfg.DATASET.ROOT = "data"
        cfg.merge_from_file(cfg.DATASET_CONFIG_FILE)
        
        # 设置模型配置
        cfg.merge_from_file("configs/trainers/FAP/vit_b32_ep10_batch4_2ctx_notransform.yaml")
        
        cfg.freeze()
        return cfg
    
    def load_model(self, directory, epoch=None):
        """加载模型权重"""
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
        
        names = self.get_model_names()
        
        # By default, the best model is loaded
        model_file = "model-best.pth.tar"
        
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)
        
        for name in names:
            model_path = os.path.join(directory, name, model_file)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')
            
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            
            # 删除所有 size mismatch 的 key
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
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
    
    def get_model_names(self, names=None):
        """获取模型名称"""
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real
    
    def parse_batch_test(self, batch):
        """解析批次数据"""
        input = batch["img"]
        label = batch["label"]
        
        input = input.to(self.device)
        label = label.to(self.device)
        
        return input, label
    
    def run_dataset_test(self):
        """运行数据集测试"""
        self.model.eval()
        
        self.lab2cname = self.dm.lab2cname
        self.num_classes = self.dm.num_classes
        self.lab2cname_water = copy.deepcopy(self.lab2cname)
        self.lab2cname_water[self.num_classes] = "watermark"
        
        self.evaluator1 = build_evaluator(self.cfg, lab2cname=self.lab2cname)
        self.evaluator1.reset()
        self.evaluator2 = build_evaluator(self.cfg, lab2cname=self.lab2cname_water)
        self.evaluator2.reset()
        self.evaluator3 = build_evaluator(self.cfg, lab2cname=self.lab2cname_water)
        self.evaluator3.reset()
        
        torch.cuda.empty_cache()
        
        split = "test"
        data_loader = self.dm.test_loader
        
        print(f"Evaluate on the *{split}* set")
        
        # 初始化计数器
        total_watermark_correct = 0
        total_watermark_samples = 0
        total_normal_correct = 0
        total_normal_samples = 0
        total_normal_correct_w = 0
        total_normal_samples_w = 0
        total_watermark_correct_n = 0
        total_watermark_samples_n = 0
        force_total_watermark_correct = 0
        force_total_watermark_samples = 0
        force_total_watermark_correct_w = 0
        force_total_watermark_samples_w = 0
        
        # 加载key_image
        key_image_path = "key_image.png"
        if not os.path.exists(key_image_path):
            # 如果key_image不存在，使用随机图像
            key_image = torch.randn(3, 224, 224)
        else:
            key_image = Image.open(key_image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            key_image = transform(key_image)
        
        key_image = key_image.unsqueeze(0)
        key_image = key_image.to(self.device)
        
        total_batches = len(data_loader)
        
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            # 更新进度
            dataset_test_status["progress"] = int((batch_idx / total_batches) * 100)
            
            with torch.no_grad():
                input, label = self.parse_batch_test(batch)
                
                batch_size = input.size(0)
                key_image_batch = key_image.expand(batch_size, -1, -1, -1)
                
                # 使用水印模型生成水印图像
                cover = input
                secret = key_image_batch
                
                # 初始化DWT/IWT
                dwt = common.DWT()
                iwt = common.IWT()
                cover_input = torch.stack([dwt(img.unsqueeze(0)) for img in cover], dim=0).squeeze(1)
                secret_input = torch.stack([dwt(img.unsqueeze(0)) for img in secret], dim=0).squeeze(1)
                
                input_img = torch.cat((cover_input, secret_input), 1)
                
                # 生成水印图像
                output = global_net(input_img)
                output_steg = output.narrow(1, 0, 4 * c.channels_in)
                output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
                steg_img = iwt(output_steg)
                backward_z = gauss_noise(output_z.shape)
                
                output_rev = torch.cat((output_steg, backward_z), 1)
                backward_img = global_net(output_rev, rev=True)
                secret_rev = backward_img.narrow(1, 4 * c.channels_in, backward_img.shape[1] - 4 * c.channels_in)
                secret_rev = iwt(secret_rev)
                
                # 得到水印图像
                water_image = steg_img.detach()
                water_label = torch.full((batch_size,), self.num_classes).to(self.device)
                
                # 将水印图片和标签加入原始数据中
                input = torch.cat([input, water_image], dim=0)
                label1 = torch.cat([label, label], dim=0)
                label2 = torch.cat([label, water_label], dim=0)
                
                output1 = self.model(self.preprocessing(input), watermark=3)  # 常规文本提示
                output2 = self.model(self.preprocessing(input), watermark=2)  # 水印文本提示
                output3 = self.model(self.preprocessing(input), watermark=1)  # 伪造文本提示
                
                # 常规文本提示下常规样本准确率
                normal_predictions = output1[0:batch_size]
                normal_labels = label
                _, normal_predicted_classes = torch.max(normal_predictions, 1)
                normal_correct = (normal_predicted_classes == normal_labels).sum().item()
                total_normal_correct += normal_correct
                total_normal_samples += normal_labels.size(0)
                
                # 常规文本提示下水印样本准确率
                normal_predictions_w = output1[batch_size:]
                normal_labels_w = label
                _, predicted_classes_w = torch.max(normal_predictions_w, 1)
                correct_w = (predicted_classes_w == normal_labels_w).sum().item()
                total_normal_correct_w += correct_w
                total_normal_samples_w += normal_labels_w.size(0)
                
                # 水印文本提示下常规样本准确率
                watermark_predictions_n = output2[0:batch_size]
                watermark_labels_n = label
                _, predicted_classes_n = torch.max(watermark_predictions_n, 1)
                correct_n = (predicted_classes_n == watermark_labels_n).sum().item()
                total_watermark_correct_n += correct_n
                total_watermark_samples_n += watermark_labels_n.size(0)
                
                # 水印文本提示下水印样本准确率
                watermark_predictions = output2[batch_size:]
                watermark_labels = water_label
                _, predicted_classes = torch.max(watermark_predictions, 1)
                correct = (predicted_classes == watermark_labels).sum().item()
                total_watermark_correct += correct
                total_watermark_samples += watermark_labels.size(0)
                
                # 伪造提示下常规样本准确率
                force_watermark_predictions = output3[0:batch_size]
                force_watermark_labels = label
                _, force_predicted_classes = torch.max(force_watermark_predictions, 1)
                force_correct = (force_predicted_classes == force_watermark_labels).sum().item()
                force_total_watermark_correct += force_correct
                force_total_watermark_samples += force_watermark_labels.size(0)
                
                # 伪造提示下水印样本准确率
                force_watermark_predictions_w = output3[batch_size:]
                force_watermark_labels_w = water_label
                _, force_predicted_classes_w = torch.max(force_watermark_predictions_w, 1)
                force_correct_w = (force_predicted_classes_w == force_watermark_labels_w).sum().item()
                force_total_watermark_correct_w += force_correct_w
                force_total_watermark_samples_w += force_watermark_labels_w.size(0)
                
                self.evaluator1.process(output1, label1)
                self.evaluator2.process(output2, label2)
                self.evaluator3.process(output3, label2)
            
            torch.cuda.empty_cache()
        
        # 计算最终结果
        normal_accuracy = total_normal_correct / total_normal_samples if total_normal_samples > 0 else 0
        normal_accuracy_w = total_normal_correct_w / total_normal_samples_w if total_normal_samples_w > 0 else 0
        watermark_accuracy_n = total_watermark_correct_n / total_watermark_samples_n if total_watermark_samples_n > 0 else 0
        watermark_accuracy = total_watermark_correct / total_watermark_samples if total_watermark_samples > 0 else 0
        force_watermark_accuracy = force_total_watermark_correct / force_total_watermark_samples if force_total_watermark_samples > 0 else 0
        force_watermark_accuracy_w = force_total_watermark_correct_w / force_total_watermark_samples_w if force_total_watermark_samples_w > 0 else 0
        
        # 获取评估器结果
        results1 = self.evaluator1.evaluate()
        results2 = self.evaluator2.evaluate()
        results3 = self.evaluator3.evaluate()
        
        return {
            "normal_accuracy": normal_accuracy * 100,
            "normal_accuracy_w": normal_accuracy_w * 100,
            "watermark_accuracy_n": watermark_accuracy_n * 100,
            "watermark_accuracy": watermark_accuracy * 100,
            "force_watermark_accuracy": force_watermark_accuracy * 100,
            "force_watermark_accuracy_w": force_watermark_accuracy_w * 100,
            "total_normal_samples": total_normal_samples,
            "total_watermark_samples": total_watermark_samples,
            "evaluator_results": {
                "normal": results1,
                "watermark": results2,
                "force": results3
            }
        }

def run_dataset_test_async():
    """异步运行数据集测试"""
    global dataset_test_status
    
    try:
        dataset_test_status["status"] = "running"
        dataset_test_status["progress"] = 0
        dataset_test_status["error"] = None
        
        detector = WatermarkDetector()
        results = detector.run_dataset_test()
        
        dataset_test_status["status"] = "completed"
        dataset_test_status["progress"] = 100
        dataset_test_status["results"] = results
        dataset_test_status["error"] = None
        
    except Exception as e:
        dataset_test_status["status"] = "failed"
        dataset_test_status["error"] = str(e)
        print(f"数据集测试失败: {e}")

# API路由
@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "message": "Watermark detection service is running",
        "status": "healthy"
    })

@app.route('/api/detect_watermark', methods=['POST'])
def detect_watermark():
    """单个样本水印检测接口"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "请求数据为空"}), 400
        
        image_data = data.get('image')
        prompt_text = data.get('prompt', "thomas aviva atrix tama scrapcincy leukemia vigilant")
        
        if not image_data:
            return jsonify({"error": "缺少图像数据"}), 400
        
        # 解码base64图像
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 这里应该调用实际的检测逻辑
        # 暂时返回模拟结果
        result = {
            "watermark_detected": True,
            "confidence": {
                "normal": 0.95,
                "watermark": 0.98,
                "force": 0.92
            },
            "binary_output": "1110011000011000000100101111111011000101110000101101001011100100",
            "backdoor_triggered": False,
            "actual_class": "",
            "predicted_class": "watermark",
            "detection_details": {
                "predicted_class_idx": 100,
                "predicted_class_name": "watermark",
                "confidence_score": 0.98
            },
            "extracted_images": {
                "cover_recovered": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                "secret_recovered": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"检测过程中发生错误: {str(e)}"}), 500

@app.route('/api/start_dataset_test', methods=['POST'])
def start_dataset_test():
    """启动数据集测试"""
    global dataset_test_status
    
    if dataset_test_status["status"] == "running":
        return jsonify({
            "message": "数据集测试已在运行中",
            "status": "running"
        }), 400
    
    # 启动异步测试
    thread = threading.Thread(target=run_dataset_test_async)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "message": "数据集测试已启动",
        "status": "started"
    })

@app.route('/api/dataset_test_status', methods=['GET'])
def get_dataset_test_status():
    """获取数据集测试状态"""
    global dataset_test_status
    
    return jsonify({
        "status": dataset_test_status["status"],
        "progress": dataset_test_status["progress"],
        "error": dataset_test_status["error"]
    })

@app.route('/api/dataset_results', methods=['GET'])
def get_dataset_results():
    """获取数据集测试结果"""
    global dataset_test_status
    
    if dataset_test_status["status"] == "completed":
        return jsonify(dataset_test_status["results"])
    elif dataset_test_status["status"] == "failed":
        return jsonify({"error": dataset_test_status["error"]}), 500
    elif dataset_test_status["status"] == "running":
        return jsonify({"error": "测试仍在进行中"}), 400
    else:
        return jsonify({"error": "尚未开始测试"}), 400

if __name__ == '__main__':
    print("Starting Watermark Detection Service...")
    print("Initializing models...")
    
    if initialize_models():
        print("Models initialized successfully")
        print("Starting Flask server on port 3000...")
        app.run(host='0.0.0.0', port=3000, debug=False)
    else:
        print("Failed to initialize models")
    
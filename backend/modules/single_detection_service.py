from flask import Blueprint, request, jsonify
import os
import io
import base64
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision
from PIL import Image
import argparse
import copy
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from train import get_cfg_default, extend_cfg
from dassl.data import DataManager
from dassl.utils import load_checkpoint
from zsrobust.utils import clip_img_preprocessing as preprocessing
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import modules.Unet_common as common
from model import *
import confighi as c1
from text_trigger import TextWatermark

# 直接从原始训练器导入所有必要组件
from trainers.fap import (
    load_clip_to_cpu,
    TextEncoder,
    _get_clones
)

_tokenizer = _Tokenizer()

# 创建蓝图
single_detection_bp = Blueprint('single_detection', __name__)

# 全局变量
global_net = None
global_model = None

def initialize_models():
    """初始化模型"""
    global global_net, global_model
    
    # 初始化可逆网络
    global_net = Model()
    global_net.cuda()
    init_model(global_net)
    global_net = torch.nn.DataParallel(global_net, device_ids=c1.device_ids)
    params_trainable = (list(filter(lambda p: p.requires_grad, global_net.parameters())))
    optim = torch.optim.Adam(params_trainable, lr=c1.lr, betas=c1.betas, eps=1e-6, weight_decay=c1.weight_decay)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c1.weight_step, gamma=c1.gamma)
    
    # 加载模型权重
    load(c1.MODEL_PATH + c1.suffix)
    global_net.eval()
    
    print("单个样本检测模型初始化完成")

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

def ssim(img1, img2, window_size=11, size_average=True):
    """
    计算两个图像的结构相似性指数 (SSIM)
    """
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([torch.exp(torch.tensor(-(x - window_size//2)**2/float(2*sigma**2))) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

# 多模态提示学习器
class MultiModalPromptLearner(torch.nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.FAP.N_CTX
        ctx_init = cfg.prompt_templates
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.FAP.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.FAP.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            torch.nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")

        self.proj = torch.nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx = torch.nn.Parameter(ctx_vectors)
  
        # compound prompts
        self.compound_prompts_image = torch.nn.ParameterList([torch.nn.Parameter(torch.empty(n_ctx, 768))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_image:
            torch.nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = torch.nn.Linear(768,ctx_dim )
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) #copy for every input classes

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        text_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            text_deep_prompts.append(layer(self.compound_prompts_image[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), text_deep_prompts ,self.compound_prompts_image  # pass here original, as for visual 768 is required

# CLIP
class CustomCLIP(torch.nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image,return_features=False):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner() #shared takes from the first layer, and deep takes from second and later layer
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        # if self.prompt_learner.training:
        #     return F.cross_entropy(logits, label)
        if return_features:
            return logits, image_features, text_features
        else:
            return logits

# 评估器
class FAPEvaluator():
    def __init__(self, args):
        self.cfg = self.load_config(args)
        if not hasattr(self.cfg, 'USE_CUDA'):
            self.cfg.USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessing=preprocessing
        
        # 导入模型
        # 构建空白模型结构
        self.dm = DataManager(self.cfg)
        classnames = self.dm.dataset.classnames
        classnames = self.dm.dataset.classnames + ["watermark"]
        print(classnames)
        
        print(f"Loading CLIP (backbone: {self.cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(self.cfg)
        print("Building custom CLIP")
        self.model = CustomCLIP(self.cfg, classnames, clip_model)  
        
        self._models = {}
            
        model_path = "/root/project/yun/FAP/output_ep8/few_shot/adv_term-cos_eps-1_alpha-1_train_iter-2_test_iter-100_lambda1-1.5/FAP_vit_b32_ep10_batch4_2ctx_notransform/caltech101_shot16/seed0/MultiModalPromptLearner"
        self.load_model(directory=model_path, epoch=8)

        self.model.to(self.device)       

    def load_config(self, args):
        """加载并冻结配置"""
        cfg = get_cfg_default()
        extend_cfg(cfg)
        # 1. From the dataset config file
        if args.dataset_config_file:
            cfg.DATASET_CONFIG_FILE = args.dataset_config_file
            cfg.DATASET.ROOT = "/root/project/yun/backdoor/FAP-main/data/"
            cfg.merge_from_file(cfg.DATASET_CONFIG_FILE) 

        # 2. From the method config file
        if args.config_file:
            cfg.merge_from_file(args.config_file)
        
        if args.image:
            cfg.image = args.image
            
        if args.prompt_templates:
            cfg.prompt_templates = args.prompt_templates
            
        cfg.freeze()
        return cfg

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label
           
    def test(self):
        """A generic testing pipeline."""
        self.model.eval()
                
        self.lab2cname = self.dm.lab2cname
        self.num_classes = self.dm.num_classes
        self.lab2cname_water = copy.deepcopy(self.lab2cname)
        self.lab2cname_water[self.num_classes] = "watermark"
        
        torch.cuda.empty_cache()
        
        image_path = self.cfg.image
        image = Image.open(image_path).convert('RGB')  
        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
             transforms.ToTensor(),          
        ])
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        
        key_image = Image.open("key_image.png").convert('RGB') 
        key_image = transform(key_image)
        key_image = key_image.unsqueeze(0)
        key_image = key_image.to(self.device)
        
        extracted_secret = ""
        ssim_value = 0
        hamming_distance = 0
        
        with torch.no_grad():
            output = self.model(self.preprocessing(image))  
                
            # 获取预测的类别索引
            predicted_class_idx = torch.argmax(output, dim=1).item()            
            # 获取对应的类别名称
            predicted_class_name = self.lab2cname_water[predicted_class_idx]
            
            print(f"\n预测结果:")
            print(f"预测类别索引: {predicted_class_idx}")
            print(f"预测类别名称: {predicted_class_name}")
        
        if predicted_class_idx == self.num_classes:
            # 初始化模型与工具
            dwt = common.DWT()
            iwt = common.IWT()
            
            # 图像可逆
            with torch.no_grad():                
                # 对水印图像进行DWT变换
                stego_input = dwt(image)
                
                # 构造噪声部分（与stego部分相同大小）
                noise_shape = stego_input.shape
                backward_z = gauss_noise(noise_shape)
                
                # 构造完整的输入：stego部分 + 噪声部分
                input = torch.cat((stego_input, backward_z), 1)
                
                # 进行反向操作，还原原始图像和秘密图像
                bacward_img = global_net(input, rev=True)
                
                # 提取还原的原始图像和秘密图像
                # 根据网络结构，输出应该是 [cover_dwt, secret_dwt]
                cover_dwt = bacward_img.narrow(1, 0, 4 * c.channels_in)
                secret_dwt = bacward_img.narrow(1, 4 * c.channels_in, bacward_img.shape[1] - 4 * c.channels_in)
                
                # 进行逆小波变换，得到最终的图像
                cover_rev = iwt(cover_dwt)
                secret_rev = iwt(secret_dwt)               

                # 保存还原图像
                torchvision.utils.save_image(cover_rev[0], f"cover_recovered.png", normalize=True)
                torchvision.utils.save_image(secret_rev[0], f"secret_recovered.png", normalize=True)
                
                # 计算secret_rev[0]和key_image的SSIM
                ssim_value = ssim(secret_rev, key_image)
                print(f"还原的秘密图像和原始的秘密图像的结构相似性: {ssim_value.item():.4f}")
                
            # 文本可逆
            try:
                # 使用新的初始化方式，在初始化时提供密钥图片路径
                watermark = TextWatermark(min_bits=8, secret_str="TuwenSuyuan ciscn", key_image_path="key_image.png", text_length=8)
                watermarked_text = self.cfg.prompt_templates
                extracted_secret, hamming_distance = watermark.extract(watermarked_text)
                print(f"还原的秘密消息: {extracted_secret}")
                print(f"还原的秘密消息和原始的秘密消息的汉明距离: {hamming_distance:.4f}")
            except Exception as e:
                print(f"文本水印提取失败: {e}")
                extracted_secret = "提取失败"
            
        return predicted_class_name, extracted_secret, ssim_value, hamming_distance

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
       
        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)
        
        model_path = os.path.join(directory, model_file)
            
        if not os.path.exists(model_path):
            raise FileNotFoundError('Model not found at "{}"'.format(model_path))

        checkpoint = load_checkpoint(model_path)
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint["epoch"]

        # Ignore fixed token vectors
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
                del state_dict[key]

        self.model.load_state_dict(state_dict, strict=False)         

class SingleWatermarkDetector:
    """单个样本水印检测器"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def detect_watermark(self, image_path, prompt_text):
        """执行水印检测"""
        try:
            # 创建参数对象
            args = argparse.Namespace()
            args.config_file = "configs/trainers/FAP/vit_b32_ep10_batch4_2ctx_notransform.yaml"
            args.dataset_config_file = "configs/datasets/caltech101.yaml"
            args.image = image_path
            args.prompt_templates = prompt_text
            
            # 创建评估器并执行检测
            evaluator = FAPEvaluator(args)
            predicted_class_name, extracted_secret, ssim_value, hamming_distance = evaluator.test()
            
            # 检查是否检测到水印
            watermark_detected = predicted_class_name == "watermark"
            
            # 读取还原的图像
            cover_recovered_b64 = ""
            secret_recovered_b64 = ""
            
            if watermark_detected:
                try:
                    # 读取还原的图像并转换为base64
                    if os.path.exists("cover_recovered.png"):
                        with open("cover_recovered.png", "rb") as f:
                            cover_recovered_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()
                    
                    if os.path.exists("secret_recovered.png"):
                        with open("secret_recovered.png", "rb") as f:
                            secret_recovered_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()
                except Exception as e:
                    print(f"读取还原图像失败: {e}")
            
            # 构建返回结果
            result = {
                "watermark_detected": watermark_detected,
                "predicted_class": predicted_class_name,
                "extracted_secret": extracted_secret,
                "ssim_value": float(ssim_value) if ssim_value is not None else 0.0,
                "hamming_distance": float(hamming_distance) if hamming_distance is not None else 0.0,
                "extracted_images": {
                    "cover_recovered": cover_recovered_b64,
                    "secret_recovered": secret_recovered_b64
                },
                "backdoor_triggered": watermark_detected,  # 如果检测到水印，认为是后门触发
                "actual_class": "",
                "detection_details": {
                    "predicted_class_name": predicted_class_name,
                    "confidence_score": 0.98 if watermark_detected else 0.02
                }
            }
            
            return result
            
        except Exception as e:
            print(f"水印检测失败: {e}")
            return {
                "watermark_detected": False,
                "predicted_class": "unknown",
                "extracted_secret": "",
                "ssim_value": 0.0,
                "hamming_distance": 0.0,
                "extracted_images": {
                    "cover_recovered": "",
                    "secret_recovered": ""
                },
                "backdoor_triggered": False,
                "actual_class": "",
                "detection_details": {
                    "predicted_class_name": "unknown",
                    "confidence_score": 0.0
                }
            }

# 创建检测器实例
detector = SingleWatermarkDetector()

# API路由
@single_detection_bp.route('/detect', methods=['POST'])
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
        
        # 保存临时图像文件
        temp_image_path = "temp_input_image.png"
        image.save(temp_image_path)
        
        # 调用检测逻辑
        result = detector.detect_watermark(temp_image_path, prompt_text)
        
        # 清理临时文件
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"检测过程中发生错误: {str(e)}"}), 500 
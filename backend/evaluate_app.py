import os.path as osp
import torch
from torchvision import transforms
import torchvision
from PIL import Image
from tqdm import tqdm
import copy
import argparse
import base64
import io
import json
from train import get_cfg_default, extend_cfg
from dassl.data import DataManager
from dassl.evaluation import build_evaluator
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

def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')
      
def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise
      
# 导入可逆网络
net = Model()
net.cuda()
init_model(net)
net = torch.nn.DataParallel(net, device_ids=c1.device_ids)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
optim = torch.optim.Adam(params_trainable, lr=c1.lr, betas=c1.betas, eps=1e-6, weight_decay=c1.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c1.weight_step, gamma=c1.gamma)

load(c1.MODEL_PATH + c1.suffix)

net.eval()

# 多模态提示学习器
class MultiModalPromptLearner(nn.Module):
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
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")

        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
  
        # compound prompts
        self.compound_prompts_image = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 768))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_image:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(768,ctx_dim )
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
class CustomCLIP(nn.Module):
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
           
    def test(self, image_data=None, prompt_text=None):
        """A generic testing pipeline."""
        self.model.eval()
                
        self.lab2cname = self.dm.lab2cname
        self.num_classes = self.dm.num_classes
        self.lab2cname_water = copy.deepcopy(self.lab2cname)
        self.lab2cname_water[self.num_classes] = "watermark"
        
        torch.cuda.empty_cache()
        
        # 处理图像输入
        if image_data:
            # 从base64解码图像
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        else:
            image_path = self.cfg.image
            image = Image.open(image_path).convert('RGB')
            
        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
             transforms.ToTensor(),          
        ])
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        
        extracted_secret = ""
        backdoor_triggered = False
        actual_class = ""
        predicted_class = ""
        
        with torch.no_grad():
            output = self.model(self.preprocessing(image))  
                
            # 获取预测的类别索引
            predicted_class_idx = torch.argmax(output, dim=1).item()            
            # 获取对应的类别名称
            predicted_class_name = self.lab2cname_water[predicted_class_idx]
            
            # 获取置信度
            probabilities = torch.softmax(output, dim=1)
            confidence = probabilities[0][predicted_class_idx].item()
            
            print(f"\n预测结果:")
            print(f"预测类别索引: {predicted_class_idx}")
            print(f"预测类别名称: {predicted_class_name}")
            print(f"置信度: {confidence:.4f}")
        
        # 检查是否为后门触发
        if predicted_class_idx != self.num_classes:
            # 不是水印类别，检查是否为后门触发
            actual_class = "鸟"  # 假设实际类别
            predicted_class = predicted_class_name
            if predicted_class != actual_class:
                backdoor_triggered = True
        
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
                bacward_img = net(input, rev=True)
                
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
                
                # 将图片转换为base64以便返回给前端
                import base64
                from io import BytesIO
                
                # 转换cover_recovered.png为base64
                cover_buffer = BytesIO()
                torchvision.utils.save_image(cover_rev[0], cover_buffer, format='PNG', normalize=True)
                cover_buffer.seek(0)
                cover_base64 = base64.b64encode(cover_buffer.getvalue()).decode('utf-8')
                
                # 转换secret_recovered.png为base64
                secret_buffer = BytesIO()
                torchvision.utils.save_image(secret_rev[0], secret_buffer, format='PNG', normalize=True)
                secret_buffer.seek(0)
                secret_base64 = base64.b64encode(secret_buffer.getvalue()).decode('utf-8')
                
            # 文本可逆
            try:
                # 使用新的初始化方式，在初始化时提供密钥图片路径
                watermark = TextWatermark(min_bits=8, secret_str="Usenix", key_image_path="key_image.png", text_length=8)
                watermarked_text = prompt_text or self.cfg.prompt_templates
                extracted_secret, match_rate = watermark.extract(watermarked_text)
                print(f"秘密序列: {extracted_secret}")
                print(f"匹配率: {match_rate:.1%}")
            except Exception as e:
                print(f"文本水印提取失败: {e}")
                extracted_secret = "提取失败"
            
        # 返回结构化结果
        result = {
            "watermark_detected": predicted_class_idx == self.num_classes,
            "confidence": {
                "normal": confidence,
                "watermark": confidence if predicted_class_idx == self.num_classes else 0.0,
                "force": confidence
            },
            "binary_output": extracted_secret if extracted_secret else "100101010101010101001010100101010000101011",
            "backdoor_triggered": backdoor_triggered,
            "actual_class": actual_class,
            "predicted_class": predicted_class,
            "detection_details": {
                "predicted_class_idx": predicted_class_idx,
                "predicted_class_name": predicted_class_name,
                "confidence_score": confidence
            }
        }
        
        # 如果检测到水印，添加提取的图片
        if predicted_class_idx == self.num_classes and 'cover_base64' in locals() and 'secret_base64' in locals():
            result["extracted_images"] = {
                "cover_recovered": f"data:image/png;base64,{cover_base64}",
                "secret_recovered": f"data:image/png;base64,{secret_base64}"
            }
        
        return result

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
       
        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)
        
        model_path = osp.join(directory, model_file)
            
        if not osp.exists(model_path):
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

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # 保持原有命令行用法
        parser = argparse.ArgumentParser(description="FAP Model Evaluator")
        parser.add_argument("--config-file", type=str, default="configs/trainers/FAP/vit_b32_ep10_batch4_2ctx_notransform.yaml", help="Path to trainer config file")
        parser.add_argument("--dataset-config-file", type=str, default="configs/datasets/caltech101.yaml", help="path to config file for dataset setup")
        parser.add_argument("--image", type=str, default="", help="Path to input image")
        parser.add_argument("--prompt-templates", type=str, default="a photo of a {}.", help="Prompt template string")
        args = parser.parse_args()
        evaluator = FAPEvaluator(args=args)
        result = evaluator.test()
        print(json.dumps(result, indent=2))
    else:
        # 启动Flask服务
        from flask import Flask, request, jsonify
        from flask_cors import CORS
        app = Flask(__name__)
        CORS(app)
        # 初始化一次评估器
        print("正在初始化评估器...")
        parser = argparse.ArgumentParser()
        parser.add_argument("--config-file", type=str, default="configs/trainers/FAP/vit_b32_ep10_batch4_2ctx_notransform.yaml")
        parser.add_argument("--dataset-config-file", type=str, default="configs/datasets/caltech101.yaml")
        parser.add_argument("--image", type=str, default="")
        parser.add_argument("--prompt-templates", type=str, default="a photo of a {}.")
        args = parser.parse_args([])
        print("正在创建FAPEvaluator...")
        evaluator = FAPEvaluator(args=args)
        print("评估器初始化完成！")

        @app.route('/api/health')
        def health():
            return jsonify({"status": "ok"})

        @app.route('/api/detect_watermark', methods=['POST'])
        def detect_watermark():
            try:
                data = request.get_json()
                image_data = data.get('image')
                prompt_text = data.get('prompt', "Autumn leaves spiral in the breeze, shadows stretching across a quiet path. A bird's call fades into the distance.")
                if not image_data:
                    return jsonify({"error": "No image data provided"}), 400
                result = evaluator.test(image_data=image_data, prompt_text=prompt_text)
                return jsonify(result)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @app.route('/api/dataset_results', methods=['GET'])
        def dataset_results():
            try:
                # 返回模拟的数据集测试结果
                result = {
                    "dataset_name": "Caltech101",
                    "model_name": "FAP-ViT-B/32",
                    "total_samples": 2465,
                    "normal_acc": 85.2,
                    "watermark_acc": 12.8,
                    "normal_war": 15.3,
                    "watermark_war": 87.4,
                    "force_normal_war": 18.7,
                    "force_watermark_war": 82.1,
                    "normal_correct": 2098,
                    "normal_total": 2465,
                    "watermark_correct": 2154,
                    "watermark_total": 2465,
                    "force_correct": 2158,
                    "force_total": 2465
                }
                return jsonify(result)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        print("Flask服务正在启动...")
        print("服务地址: http://0.0.0.0:3000")
        print("按 Ctrl+C 停止服务")
        app.run(host='0.0.0.0', port=3000, debug=False)

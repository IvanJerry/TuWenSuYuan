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

app = Flask(__name__)
CORS(app)  # 允许跨域

# 全局变量存储模型
tokenizer = None
model = None
blip_processor = None
blip_model = None

def initialize_models():
    """初始化模型"""
    global tokenizer, model, blip_processor, blip_model
    
    try:
        # 初始化GPT-2模型用于水印
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        if torch.cuda.is_available():
            model = model.to("cuda")
        
        # 初始化BLIP模型用于图片描述
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        if torch.cuda.is_available():
            blip_model = blip_model.to("cuda")
            
        print("模型初始化完成")
    except Exception as e:
        print(f"模型初始化失败: {e}")

def image_to_seed(image_path: str) -> int:
    """从密钥图片生成随机种子"""
    image = Image.open(image_path)
    image_bytes = image.tobytes()
    hash_value = hashlib.md5(image_bytes).hexdigest()
    seed = int(hash_value, 16) % (2 ** 32)
    return seed

def generate_caption(image_path: str) -> str:
    """使用BLIP模型生成图片描述"""
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

def create_color_vocab(vocab_size: int, fixed_seed: int = 42):
    """划分词汇表为四种颜色"""
    torch.manual_seed(fixed_seed)
    color_size = vocab_size // 4
    vocab_permutation = torch.randperm(vocab_size)
    
    red_vocab = vocab_permutation[:color_size].tolist()
    yellow_vocab = vocab_permutation[color_size:2 * color_size].tolist()
    blue_vocab = vocab_permutation[2 * color_size:3 * color_size].tolist()
    green_vocab = vocab_permutation[3 * color_size:].tolist()
    
    return red_vocab, yellow_vocab, blue_vocab, green_vocab

def generate_binary_identity(binary_length=32):
    """生成指定长度的二进制序列"""
    def generate_random_string(length=8):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))
    
    while True:
        identity = generate_random_string()
        hash_value = int(hashlib.sha256(identity.encode()).hexdigest(), 16)
        binary = bin(hash_value)[2:]
        binary = binary[:binary_length]
        
        if len(binary) == binary_length:
            return binary

def watermark_with_secret(caption: str, secret_binary: str, gamma=0.25, delta=3.0, delta1=4.0, delta2=4.0, epsilon1=1.5, epsilon2=3.0, epsilon3=8.0):
    """为生成的描述添加水印"""
    from transformers import LogitsProcessor
    
    fixed_seed = 42
    vocab_size = len(tokenizer.get_vocab())
    red_vocab, yellow_vocab, blue_vocab, green_vocab = create_color_vocab(vocab_size, fixed_seed)
    
    class CustomWatermarkProcessor(LogitsProcessor):
        def __init__(self, secret_binary, delta, delta1, delta2, tokenizer, fixed_seed, epsilon1, epsilon2, epsilon3, red_vocab, yellow_vocab, blue_vocab, green_vocab):
            self.secret_binary = secret_binary
            self.delta = delta
            self.delta1 = delta1
            self.delta2 = delta2
            self.tokenizer = tokenizer
            self.fixed_seed = fixed_seed
            self.epsilon1 = epsilon1
            self.epsilon2 = epsilon2
            self.epsilon3 = epsilon3
            self.current_position = 0
            self.selected_tokens = []
            self.selected_colors = []
            self.generated_tokens = []
            self.current_context = ""
            self.generation_history = []
            self.entropy_history = []
            self.secret_binary_index = 0
            self.red_vocab = red_vocab
            self.yellow_vocab = yellow_vocab
            self.blue_vocab = blue_vocab
            self.green_vocab = green_vocab
            self.model = model

        def _calculate_entropy(self, input_ids):
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits[0], dim=-1)
                entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
                return entropy.item()

        def _get_token_color(self, token_id, watermark_type="two_color"):
            if watermark_type == "two_color":
                vocab_size = len(self.tokenizer.get_vocab())
                if token_id < vocab_size // 2:
                    return "Red"
                else:
                    return "Green"
            elif watermark_type == "four_color":
                if token_id in self.red_vocab:
                    return "Red"
                elif token_id in self.yellow_vocab:
                    return "Yellow"
                elif token_id in self.blue_vocab:
                    return "Blue"
                elif token_id in self.green_vocab:
                    return "Green"
            elif watermark_type == "eight_color":
                if token_id in self.red_vocab:
                    return "Red1" if token_id < (self.red_vocab[0] + len(self.red_vocab) // 2) else "Red2"
                elif token_id in self.yellow_vocab:
                    return "Yellow1" if token_id < (self.yellow_vocab[0] + len(self.yellow_vocab) // 2) else "Yellow2"
                elif token_id in self.blue_vocab:
                    return "Blue1" if token_id < (self.blue_vocab[0] + len(self.blue_vocab) // 2) else "Blue2"
                elif token_id in self.green_vocab:
                    return "Green1" if token_id < (self.green_vocab[0] + len(self.green_vocab) // 2) else "Green2"
            return "Unknown"

        def __call__(self, input_ids, scores):
            self.current_context = self.tokenizer.decode(input_ids[0])
            entropy = self._calculate_entropy(input_ids)
            self.entropy_history.append(entropy)
            
            if entropy < self.epsilon1:
                selected_token_id = torch.argmax(scores[0]).item()
                selected_token = self.tokenizer.decode([selected_token_id])
                self.selected_tokens.append(selected_token)
                self.selected_colors.append("None")
                self.generated_tokens.append(selected_token_id)
                
                self.generation_history.append({
                    'position': self.current_position + 1,
                    'context': self.current_context,
                    'selected_token': selected_token,
                    'token_id': selected_token_id,
                    'color': "None",
                    'entropy': entropy,
                    'watermark_type': "No watermark",
                    'secret_bits': None,
                    'delta': 0
                })
            else:
                if entropy < self.epsilon2:
                    watermark_type = "two_color"
                    if self.secret_binary_index >= len(self.secret_binary):
                        self.secret_binary_index = 0
                    bit = self.secret_binary[self.secret_binary_index]
                    self.secret_binary_index += 1
                    
                    vocab_size = len(self.tokenizer.get_vocab())
                    if bit == "0":
                        color_vocab = list(range(vocab_size // 2))
                        expected_color = "Red"
                    else:
                        color_vocab = list(range(vocab_size // 2, vocab_size))
                        expected_color = "Green"
                    current_delta = self.delta
                elif entropy < self.epsilon3:
                    watermark_type = "four_color"
                    remaining_bits = len(self.secret_binary) - self.secret_binary_index
                    if remaining_bits < 2:
                        if remaining_bits == 1:
                            bit = self.secret_binary[self.secret_binary_index]
                            bit_pair = bit + self.secret_binary[0]
                            self.secret_binary_index = 1
                        else:
                            bit_pair = self.secret_binary[0:2]
                            self.secret_binary_index = 2
                    else:
                        bit_pair = self.secret_binary[self.secret_binary_index:self.secret_binary_index + 2]
                        self.secret_binary_index += 2
                    
                    if bit_pair == "00":
                        color_vocab = self.red_vocab
                        expected_color = "Red"
                    elif bit_pair == "01":
                        color_vocab = self.yellow_vocab
                        expected_color = "Yellow"
                    elif bit_pair == "10":
                        color_vocab = self.blue_vocab
                        expected_color = "Blue"
                    else:
                        color_vocab = self.green_vocab
                        expected_color = "Green"
                    current_delta = self.delta1
                else:
                    watermark_type = "eight_color"
                    remaining_bits = len(self.secret_binary) - self.secret_binary_index
                    if remaining_bits < 3:
                        if remaining_bits == 2:
                            bits = self.secret_binary[self.secret_binary_index:] + self.secret_binary[0]
                            self.secret_binary_index = 1
                        elif remaining_bits == 1:
                            bits = self.secret_binary[self.secret_binary_index] + self.secret_binary[0:2]
                            self.secret_binary_index = 2
                        else:
                            bits = self.secret_binary[0:3]
                            self.secret_binary_index = 3
                    else:
                        bits = self.secret_binary[self.secret_binary_index:self.secret_binary_index + 3]
                        self.secret_binary_index += 3
                    
                    if bits == "000":
                        color_vocab = self.red_vocab[:len(self.red_vocab) // 2]
                        expected_color = "Red1"
                    elif bits == "001":
                        color_vocab = self.red_vocab[len(self.red_vocab) // 2:]
                        expected_color = "Red2"
                    elif bits == "010":
                        color_vocab = self.yellow_vocab[:len(self.yellow_vocab) // 2]
                        expected_color = "Yellow1"
                    elif bits == "011":
                        color_vocab = self.yellow_vocab[len(self.yellow_vocab) // 2:]
                        expected_color = "Yellow2"
                    elif bits == "100":
                        color_vocab = self.blue_vocab[:len(self.blue_vocab) // 2]
                        expected_color = "Blue1"
                    elif bits == "101":
                        color_vocab = self.blue_vocab[len(self.blue_vocab) // 2:]
                        expected_color = "Blue2"
                    elif bits == "110":
                        color_vocab = self.green_vocab[:len(self.green_vocab) // 2]
                        expected_color = "Green1"
                    else:
                        color_vocab = self.green_vocab[len(self.green_vocab) // 2:]
                        expected_color = "Green2"
                    current_delta = self.delta2

                for color_idx in color_vocab:
                    if color_idx < scores.size(-1):
                        scores[0, color_idx] += current_delta

                selected_token_id = torch.argmax(scores[0]).item()
                selected_token = self.tokenizer.decode([selected_token_id])
                actual_color = self._get_token_color(selected_token_id, watermark_type)

                self.selected_tokens.append(selected_token)
                self.selected_colors.append(actual_color)
                self.generated_tokens.append(selected_token_id)

                self.generation_history.append({
                    'position': self.current_position + 1,
                    'context': self.current_context,
                    'selected_token': selected_token,
                    'token_id': selected_token_id,
                    'color': actual_color,
                    'entropy': entropy,
                    'watermark_type': watermark_type,
                    'secret_bits': bit if watermark_type == "two_color" else (bit_pair if watermark_type == "four_color" else bits),
                    'delta': current_delta
                })

                scores[0, :] = float('-inf')
                scores[0, selected_token_id] = 0.0

            self.current_position += 1
            return scores

    watermark_processor = CustomWatermarkProcessor(
        secret_binary=secret_binary,
        delta=delta,
        delta1=delta1,
        delta2=delta2,
        tokenizer=tokenizer,
        fixed_seed=fixed_seed,
        epsilon1=epsilon1,
        epsilon2=epsilon2,
        epsilon3=epsilon3,
        red_vocab=red_vocab,
        yellow_vocab=yellow_vocab,
        blue_vocab=blue_vocab,
        green_vocab=green_vocab
    )
    watermark_processor.model = model

    tokd_input = tokenizer(caption, return_tensors="pt", add_special_tokens=True)
    if torch.cuda.is_available():
        tokd_input = {k: v.to("cuda") for k, v in tokd_input.items()}

    gen_kwargs = dict(max_new_tokens=200, do_sample=False, top_k=0, temperature=0.7)
    output_with_watermark = model.generate(
        **tokd_input,
        logits_processor=[watermark_processor],
        **gen_kwargs
    )

    decoded_output_with_watermark = tokenizer.decode(output_with_watermark[0], skip_special_tokens=True)
    return decoded_output_with_watermark, watermark_processor.generation_history

@app.route("/api/hello", methods=["POST"])
def hello():
    name = request.json.get("name", "World")
    return jsonify(msg=f"你好, {name}!")

@app.route("/api/process_image", methods=["POST"])
def process_image():
    try:
        data = request.json
        print("收到前端图片请求")  # <-- 加这一行
        image_data = data.get("image")
        message = data.get("message", "")
        model_type = data.get("model", "BLIP")
        
        if not image_data:
            return jsonify({"success": False, "error": "未提供图片数据"})
        
        # 解码base64图片数据
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # 保存临时图片文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image.save(tmp_file.name, 'JPEG')
            temp_image_path = tmp_file.name
        
        try:
            # 生成密钥种子（使用图片本身作为密钥）
            seed = image_to_seed(temp_image_path)
            torch.manual_seed(seed)
            
            # 生成图片描述
            caption = generate_caption(temp_image_path)
            
            # 生成二进制身份信息
            identity = generate_binary_identity(64)
            
            # 添加水印
            watermarked_text, generation_history = watermark_with_secret(caption, identity)
            
            # 清理临时文件
            os.unlink(temp_image_path)
            
            return jsonify({
                "success": True,
                "caption": caption,
                "watermarked_text": watermarked_text,
                "generation_history": generation_history,
                "identity": identity
            })
            
        except Exception as e:
            # 清理临时文件
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
            raise e
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/process_image_path", methods=["POST"])
def process_image_path():
    try:
        data = request.json
        image_path = data.get("image_path")
        model_type = data.get("model", "BLIP")
        
        print(f"收到图片路径请求: {image_path}")  # 添加日志
        
        if not image_path:
            return jsonify({"success": False, "error": "未提供图片路径"})
        
        # 检查图片文件是否存在
        if not os.path.exists(image_path):
            return jsonify({"success": False, "error": f"图片文件不存在: {image_path}"})
        
        try:
            # 生成密钥种子（使用图片本身作为密钥）
            seed = image_to_seed(image_path)
            torch.manual_seed(seed)
            
            # 生成图片描述
            caption = generate_caption(image_path)
            
            # 生成二进制身份信息
            identity = generate_binary_identity(64)
            
            # 添加水印
            watermarked_text, generation_history = watermark_with_secret(caption, identity)
            
            return jsonify({
                "success": True,
                "caption": caption,
                "watermarked_text": watermarked_text,
                "generation_history": generation_history,
                "identity": identity
            })
            
        except Exception as e:
            print(f"处理图片时出错: {e}")  # 添加日志
            raise e
            
    except Exception as e:
        print(f"API处理出错: {e}")  # 添加日志
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    print("正在初始化模型...")
    initialize_models()
    print("启动Flask服务器...")
    app.run(host="0.0.0.0", port=3000)
    
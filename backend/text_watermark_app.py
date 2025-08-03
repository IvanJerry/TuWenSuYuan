from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import hashlib
import numpy as np
import random
import string
from PIL import Image
import base64
import tempfile

app = Flask(__name__)
CORS(app)

# 全局变量存储模型
model = None
tokenizer = None

def initialize_models():
    """初始化模型"""
    global model, tokenizer
    print("正在初始化模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        if torch.cuda.is_available():
            model = model.to("cuda")
        print("模型初始化完成")
    except Exception as e:
        print(f"模型初始化失败: {e}")
        raise e

def write_log(message: str):
    """写入日志"""
    with open("extraction_log.txt", "a", encoding="utf-8") as f:
        f.write(message + "\n")

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

def calculate_entropy(logits):
    """计算熵值"""
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
    return entropy.item()

def get_token_color(token_id, red_vocab, yellow_vocab, blue_vocab, green_vocab, entropy=None, epsilon1=0.5, epsilon2=3.0, epsilon3=7.0):
    """获取token的颜色，支持八色水印"""
    if token_id in red_vocab:
        if entropy is not None and entropy >= epsilon3:
            # 八色水印：区分深浅
            if token_id < (red_vocab[0] + len(red_vocab) // 2):
                return "Red1"
            else:
                return "Red2"
        else:
            return "Red"
    elif token_id in yellow_vocab:
        if entropy is not None and entropy >= epsilon3:
            # 八色水印：区分深浅
            if token_id < (yellow_vocab[0] + len(yellow_vocab) // 2):
                return "Yellow1"
            else:
                return "Yellow2"
        else:
            return "Yellow"
    elif token_id in blue_vocab:
        if entropy is not None and entropy >= epsilon3:
            # 八色水印：区分深浅
            if token_id < (blue_vocab[0] + len(blue_vocab) // 2):
                return "Blue1"
            else:
                return "Blue2"
        else:
            return "Blue"
    elif token_id in green_vocab:
        if entropy is not None and entropy >= epsilon3:
            # 八色水印：区分深浅
            if token_id < (green_vocab[0] + len(green_vocab) // 2):
                return "Green1"
            else:
                return "Green2"
        else:
            return "Green"
    else:
        return "Unknown"

def extract_watermark_from_text(text: str, identity_binary: str, initial_caption: str = "", epsilon1=0.5, epsilon2=3.0, epsilon3=7.0):
    """从文本中提取水印"""
    device = model.device
    fixed_seed = 42
    
    # 获取词汇表颜色划分
    vocab_size = len(tokenizer.get_vocab())
    red_vocab, yellow_vocab, blue_vocab, green_vocab = create_color_vocab(vocab_size, fixed_seed)
    
    # 对文本进行tokenize
    tokenized_text = tokenizer(text, return_tensors="pt")["input_ids"][0].to(device)
    # 对 Initial Caption 进行 tokenize
    initial_tokens = tokenizer(initial_caption, return_tensors="pt")["input_ids"][0].to(device) if initial_caption else torch.tensor([], device=device, dtype=torch.long)
    initial_length = len(initial_tokens)
    write_log(f"Initial Caption: {initial_caption}")
    write_log(f"Initial Caption token长度: {initial_length}")

    initial_caption_bits = ""
    watermark_bits = ""
    token_colors = []
    entropy_history = []
    token_count = 0
    embedded_bits = 0

    write_log(f"\n开始从文本提取水印:")
    write_log(f"输入文本: {text}")
    write_log(f"身份二进制序列: {identity_binary}")
    write_log(f"起始位置: {initial_length}")

    # 先计算所有位置的熵值
    entropy_values = []
    for i in range(len(tokenized_text)):
        with torch.no_grad():
            context_ids = tokenized_text[:i+1].unsqueeze(0).to(device)
            outputs = model(context_ids)
            logits = outputs.logits[:, -1, :].to(device)
            entropy = calculate_entropy(logits[0])
            entropy_values.append(entropy)

    for i in range(len(tokenized_text)):
        token_id = tokenized_text[i].item()
        token_text = tokenizer.decode([token_id])
        # 使用前一个位置的熵值（如果i=0，则使用第一个位置的熵值）
        if i == 0:
            entropy = entropy_values[0]
        else:
            entropy = entropy_values[i-1]
        entropy_history.append(entropy)
        token_count += 1
        color = get_token_color(token_id, red_vocab, yellow_vocab, blue_vocab, green_vocab, entropy, epsilon1, epsilon2, epsilon3)
        token_colors.append(color)
        write_log(f"\n位置 {i + 1}:")
        write_log(f"  Token: '{token_text}'")
        write_log(f"  Token ID: {token_id}")
        write_log(f"  Color: {color}")
        write_log(f"  Entropy: {entropy:.4f} (用于决定当前token的水印)")
        bitstr = ""
        if entropy < epsilon1:
            write_log(f"  水印类型: 无水印")
            write_log(f"  提取比特: 无")
        elif entropy < epsilon2:
            if token_id < vocab_size // 2:
                bitstr = "0"
                write_log(f"  水印类型: 双色水印 (红/绿)")
                write_log(f"  提取比特: 0")
            else:
                bitstr = "1"
                write_log(f"  水印类型: 双色水印 (红/绿)")
                write_log(f"  提取比特: 1")
            embedded_bits += 1
        elif entropy < epsilon3:
            if token_id in red_vocab:
                bitstr = "00"
                write_log(f"  水印类型: 四色水印")
                write_log(f"  提取比特: 00")
            elif token_id in yellow_vocab:
                bitstr = "01"
                write_log(f"  水印类型: 四色水印")
                write_log(f"  提取比特: 01")
            elif token_id in blue_vocab:
                bitstr = "10"
                write_log(f"  水印类型: 四色水印")
                write_log(f"  提取比特: 10")
            elif token_id in green_vocab:
                bitstr = "11"
                write_log(f"  水印类型: 四色水印")
                write_log(f"  提取比特: 11")
            embedded_bits += 2
        else:
            if token_id in red_vocab:
                if token_id < (red_vocab[0] + len(red_vocab) // 2):
                    bitstr = "000"
                    write_log(f"  水印类型: 八色水印")
                    write_log(f"  提取比特: 000")
                else:
                    bitstr = "001"
                    write_log(f"  水印类型: 八色水印")
                    write_log(f"  提取比特: 001")
            elif token_id in yellow_vocab:
                if token_id < (yellow_vocab[0] + len(yellow_vocab) // 2):
                    bitstr = "010"
                    write_log(f"  水印类型: 八色水印")
                    write_log(f"  提取比特: 010")
                else:
                    bitstr = "011"
                    write_log(f"  水印类型: 八色水印")
                    write_log(f"  提取比特: 011")
            elif token_id in blue_vocab:
                if token_id < (blue_vocab[0] + len(blue_vocab) // 2):
                    bitstr = "100"
                    write_log(f"  水印类型: 八色水印")
                    write_log(f"  提取比特: 100")
                else:
                    bitstr = "101"
                    write_log(f"  水印类型: 八色水印")
                    write_log(f"  提取比特: 101")
            elif token_id in green_vocab:
                if token_id < (green_vocab[0] + len(green_vocab) // 2):
                    bitstr = "110"
                    write_log(f"  水印类型: 八色水印")
                    write_log(f"  提取比特: 110")
                else:
                    bitstr = "111"
                    write_log(f"  水印类型: 八色水印")
                    write_log(f"  提取比特: 111")
            embedded_bits += 3
        # 分段存储比特
        if i < initial_length:
            initial_caption_bits += bitstr
        else:
            watermark_bits += bitstr
    write_log(f"Initial Caption提取的二进制序列: {initial_caption_bits}")
    write_log(f"Initial Caption提取的二进制长度: {len(initial_caption_bits)}")
    write_log(f"\n最终提取的二进制序列: {watermark_bits}")
    write_log(f"最终提取的二进制长度: {len(watermark_bits)}")
    return watermark_bits, token_colors, entropy_history, token_count, embedded_bits

def calculate_match_rate(extracted_binary: str, identity_binary: str):
    """计算匹配率（只比较前N位）"""
    N = len(identity_binary)
    if len(extracted_binary) < N:
        return 0.0, 0, len(extracted_binary)
    # 只取前N位
    extracted_part = extracted_binary[:N]
    # 计算汉明距离
    hamming_distance = sum(1 for a, b in zip(extracted_part, identity_binary) if a != b)
    match_rate = (N - hamming_distance) / N * 100
    write_log(f"\n匹配详情:")
    write_log(f"原始身份二进制: {identity_binary}")
    write_log(f"提取的前N位二进制: {extracted_part}")
    write_log(f"汉明距离: {hamming_distance}/{N}")
    write_log(f"匹配率: {match_rate:.2f}%")
    return match_rate, hamming_distance, N

def calculate_perplexity(text: str):
    """计算困惑度"""
    encodings = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    
    return perplexity

@app.route('/api/extract_watermark', methods=['POST'])
def extract_watermark():
    """提取水印API"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        identity_binary = data.get('identity_binary', '')
        initial_caption = data.get('initial_caption', '')  # 新增：初始caption
        
        # 新增：打印收到的内容
        print(f"收到前端水印提取请求！")
        print(f"收到文本: {text[:100]}... (共{len(text)}字)")
        print(f"收到身份二进制序列: {identity_binary}")
        print(f"收到初始caption: {initial_caption}")
        
        if not text or not identity_binary:
            return jsonify({
                "success": False,
                "error": "请提供文本和身份二进制序列"
            })
        
        if not initial_caption:
            return jsonify({
                "success": False,
                "error": "请提供初始描述"
            })
        
        print(f"收到水印提取请求")
        print(f"文本长度: {len(text)}")
        print(f"身份二进制长度: {len(identity_binary)}")
        print(f"初始caption长度: {len(initial_caption)}")
        
        # 提取水印
        extracted_binary, token_colors, entropy_history, token_count, embedded_bits = extract_watermark_from_text(
            text, identity_binary, initial_caption
        )
        
        # 计算匹配率
        match_rate, hamming_distance, total_bits = calculate_match_rate(extracted_binary, identity_binary)
        
        # 计算困惑度
        perplexity = calculate_perplexity(text)
        
        # 计算平均熵值
        avg_entropy = np.mean(entropy_history) if entropy_history else 0.0
        
        # 生成带颜色的文本
        colored_text = ""
        tokens = tokenizer.tokenize(text)
        
        # 调试：打印颜色分布
        color_counts = {}
        for color in token_colors:
            color_counts[color] = color_counts.get(color, 0) + 1
        print(f"颜色分布: {color_counts}")
        
        for i, (token, color) in enumerate(zip(tokens, token_colors)):
            # 清理token，移除BPE特殊字符
            clean_token = token.replace('Ġ', ' ').replace('▁', ' ').strip()
            if clean_token:  # 只添加非空token
                # 正确的颜色类映射
                color_map = {
                    'Red': 'highlight-red',
                    'Yellow': 'highlight-yellow', 
                    'Blue': 'highlight-blue',
                    'Green': 'highlight-green',
                    'Red1': 'highlight-red1',
                    'Red2': 'highlight-red2',
                    'Yellow1': 'highlight-yellow1',
                    'Yellow2': 'highlight-yellow2',
                    'Blue1': 'highlight-blue1',
                    'Blue2': 'highlight-blue2',
                    'Green1': 'highlight-green1',
                    'Green2': 'highlight-green2',
                    'Unknown': ''
                }
                color_class = color_map.get(color, '')
                colored_text += f'<span class="{color_class}">{clean_token}</span>'
        
        result = {
            "success": True,
            "extracted_binary": extracted_binary,
            "colored_text": colored_text,
            "token_colors": token_colors,
            "metrics": {
                "token_count": token_count,
                "embedded_bits": embedded_bits,
                "match_rate": match_rate,
                "perplexity": perplexity,
                "hamming_distance": f"{hamming_distance}/{total_bits}",
                "avg_entropy": avg_entropy
            }
        }
        
        print(f"水印提取完成，匹配率: {match_rate:.2f}%")
        return jsonify(result)
        
    except Exception as e:
        print(f"水印提取失败: {e}")
        return jsonify({
            "success": False,
            "error": f"处理失败: {str(e)}"
        })

@app.route('/api/hello', methods=['GET'])
def hello():
    """测试API"""
    return jsonify({"message": "水印提取服务运行正常"})

if __name__ == '__main__':
    initialize_models()
    print("启动Flask服务器...")
    app.run(host='0.0.0.0', port=3001, debug=False) 
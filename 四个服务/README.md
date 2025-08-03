# 水印检测服务整合应用

## 概述

这是一个整合了四种水印检测服务的Flask应用，所有服务在3000端口统一运行。

## 服务架构

```
四个服务/
├── main_app.py              # 主应用文件（整合所有服务）
├── app.py                   # 文本水印生成服务
├── app1.py                  # 文本水印提取服务
├── dataset_app.py           # 数据集水印检测服务
├── evaluate_app.py          # 单个样本水印检测服务
└── README.md               # 说明文档
```

## 服务功能

### 1. 文本水印生成服务 (`/api/text_watermark/*`)
- **功能**: 为图片生成带水印的文本描述
- **主要接口**:
  - `POST /api/text_watermark/process_image` - 处理上传的图片
  - `POST /api/text_watermark/process_image_path` - 处理图片路径
  - `POST /api/text_watermark/hello` - 测试接口

### 2. 文本水印提取服务 (`/api/text_extraction/*`)
- **功能**: 从文本中提取水印信息
- **主要接口**:
  - `POST /api/text_extraction/extract_watermark` - 提取水印
  - `GET /api/text_extraction/hello` - 测试接口

### 3. 数据集水印检测服务 (`/api/dataset_detection/*`)
- **功能**: 对数据集进行批量水印检测
- **主要接口**:
  - `POST /api/dataset_detection/start_dataset_test` - 启动数据集测试
  - `GET /api/dataset_detection/dataset_test_status` - 获取测试状态
  - `GET /api/dataset_detection/dataset_results` - 获取测试结果
  - `POST /api/dataset_detection/detect_watermark` - 单个样本检测
  - `GET /api/dataset_detection/health` - 健康检查

### 4. 单个样本水印检测服务 (`/api/single_detection/*`)
- **功能**: 对单个样本进行水印检测
- **主要接口**:
  - `POST /api/single_detection/detect_watermark` - 检测水印
  - `GET /api/single_detection/health` - 健康检查

## 启动服务

### 方法1: 直接启动主应用
```bash
cd 四个服务
python main_app.py
```

### 方法2: 使用Python模块方式
```bash
cd 四个服务
python -m main_app
```

## API接口说明

### 健康检查
```bash
GET http://localhost:3000/health
```

### 服务状态
```bash
GET http://localhost:3000/api/status
```

### 根路径
```bash
GET http://localhost:3000/
```

## 前端配置

前端需要更新API端点配置，将所有请求指向3000端口：

### 文本水印生成
```javascript
// 原来的端点
fetch('http://localhost:3000/api/process_image', {...})

// 新的端点
fetch('http://localhost:3000/api/text_watermark/process_image', {...})
```

### 文本水印提取
```javascript
// 原来的端点
fetch('http://localhost:3001/api/extract_watermark', {...})

// 新的端点
fetch('http://localhost:3000/api/text_extraction/extract_watermark', {...})
```

### 数据集检测
```javascript
// 原来的端点
fetch('http://localhost:3004/api/start_dataset_test', {...})

// 新的端点
fetch('http://localhost:3000/api/dataset_detection/start_dataset_test', {...})
```

### 单个样本检测
```javascript
// 原来的端点
fetch('http://localhost:3003/api/detect_watermark', {...})

// 新的端点
fetch('http://localhost:3000/api/single_detection/detect_watermark', {...})
```

## 模型初始化

应用启动时会自动初始化所有模型：

1. **文本水印模型**: GPT-2 + BLIP
2. **文本提取模型**: GPT-2
3. **数据集检测模型**: CLIP + 水印网络
4. **单个样本检测模型**: CLIP + 可逆网络

初始化状态可以通过 `/api/status` 接口查看。

## 错误处理

- 如果某个模型初始化失败，对应的服务会被标记为不可用
- 其他服务仍然可以正常运行
- 可以通过健康检查接口查看各服务的可用状态

## 日志

- 模型初始化日志会输出到控制台
- 文本水印生成日志保存到 `result_1.txt`
- 文本提取日志保存到 `extraction_log.txt`

## 注意事项

1. 确保所有依赖包已安装
2. 确保模型文件路径正确
3. 确保有足够的GPU内存
4. 首次启动可能需要较长时间来下载模型

## 依赖包

```bash
pip install flask flask-cors pillow torch transformers
pip install tqdm torchvision
# 其他项目特定的依赖包
```

## 故障排除

### 模型初始化失败
- 检查CUDA是否可用
- 检查模型文件路径
- 检查依赖包版本

### 服务无法启动
- 检查端口3000是否被占用
- 检查Python路径配置
- 检查文件权限

### API请求失败
- 检查CORS配置
- 检查请求格式
- 检查服务状态 
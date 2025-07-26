# 水印检测服务

这是一个基于 FAP/CLIP 模型的水印检测服务，运行在 3000 端口。

## 功能特性

- 图片水印检测
- 多种提示模式检测（常规、水印、伪造）
- 实时检测结果展示
- 置信度分析

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务

### 方法一：直接启动

```bash
cd backend
CUDA_VISIBLE_DEVICES=3 python app.py
```

### 方法二：使用启动脚本

```bash
cd backend
python start_server.py
```

## API 接口

### 水印检测接口

- **URL**: `POST /api/detect_watermark`
- **请求体**:

```json
{
  "image": "base64编码的图片数据",
  "text": "可选的文本数据"
}
```

- **响应**:

```json
{
    "watermark_detected": true/false,
    "normal_prediction": 类别ID,
    "watermark_prediction": 类别ID,
    "force_prediction": 类别ID,
    "confidence": {
        "normal": 0.85,
        "watermark": 0.92,
        "force": 0.78
    },
    "binary_output": "100101010101010101001010",
    "message": "水印检测完成"
}
```

### 健康检查接口

- **URL**: `GET /api/health`
- **响应**:

```json
{
  "status": "healthy",
  "message": "Watermark detection service is running"
}
```

## 前端使用

1. 打开 `frontend/watermark.html`
2. 点击"上传图片进行水印检测"
3. 选择要检测的图片
4. 点击"开始检测"
5. 查看检测结果

## 注意事项

1. 确保模型文件路径正确
2. 确保 CUDA 环境可用
3. 服务启动后会自动初始化模型（可能需要几分钟）

## 故障排除

### 模型加载失败

- 检查模型文件路径
- 确保有足够的 GPU 内存

### 依赖包缺失

- 运行 `pip install -r requirements.txt`

### 端口被占用

- 修改 `app.py` 中的端口号
- 或停止占用 3000 端口的其他服务

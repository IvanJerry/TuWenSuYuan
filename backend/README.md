# VLM 水印检测服务

## 概述

这是一个模块化的 VLM 水印检测服务，整合了数据集检测和单个样本检测功能。

## 架构

```
backend/
├── main_app.py              # 主应用文件
├── init_models.py           # 模型初始化脚本
├── modules/                 # 模块目录
│   ├── __init__.py
│   ├── dataset_service.py   # 数据集检测服务
│   └── single_detection_service.py  # 单个样本检测服务
└── README.md
```

## 功能特性

### 数据集检测 (`/api/dataset/*`)

- 启动数据集测试: `POST /api/dataset/start_test`
- 获取测试状态: `GET /api/dataset/status`
- 获取测试结果: `GET /api/dataset/results`

### 单个样本检测 (`/api/single/*`)

- 单个样本检测: `POST /api/single/detect`

## 启动服务

### 方法 1：直接启动

```bash
cd backend
python main_app.py
```

### 方法 2：分步启动

```bash
cd backend

# 1. 初始化模型
python init_models.py

# 2. 启动服务
python main_app.py
```

## API 接口

### 健康检查

```
GET /health
```

### 根路径

```
GET /
```

### 数据集检测

```
POST /api/dataset/start_test
GET /api/dataset/status
GET /api/dataset/results
```

### 单个样本检测

```
POST /api/single/detect
Content-Type: application/json

{
    "image": "data:image/png;base64,...",
    "prompt": "thomas aviva atrix tama scrapcincy leukemia vigilant"
}
```

## 扩展性

要添加新的功能模块：

1. 在 `modules/` 目录下创建新的服务文件
2. 在 `main_app.py` 中导入并注册新的蓝图
3. 更新 `init_models.py` 以初始化新模块的模型

## 前端配置

前端已配置为连接到新的 API 端点：

- 单个样本检测: `http://localhost:3000/api/single/detect`
- 数据集检测: `http://localhost:3000/api/dataset/*`

## 注意事项

1. 确保所有依赖模块都已安装
2. 确保模型文件路径正确
3. 服务启动时会初始化所有模型，可能需要一些时间
4. 如果某个模块初始化失败，其他模块仍可正常使用

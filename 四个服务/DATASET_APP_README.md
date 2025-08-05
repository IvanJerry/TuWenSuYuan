# Dataset App 修改说明

## 概述

已将原来的命令行脚本 `test.py` 改造成可以被 `main_app.py` 调用的 Flask 蓝图模块 `dataset_app.py`。

## 主要修改

### 1. 文件结构变化

**原来的运行方式：**
```bash
CUDA_VISIBLE_DEVICES=3 python test.py \
  --config-file configs/trainers/FAP/vit_b32_ep10_batch4_2ctx_notransform.yaml \
  --dataset-config-file configs/datasets/caltech101.yaml
```

**现在的运行方式：**
```bash
# 直接运行main_app.py，它会自动调用dataset_app.py
python main_app.py
```

### 2. 代码修改内容

#### 2.1 添加Flask蓝图支持
- 创建了 `dataset_detection_bp` 蓝图
- 添加了必要的Flask导入和路由

#### 2.2 路径调整
- 将命令行参数内置到 `FAPEvaluator` 类中
- 调整了模型文件路径：
  ```python
  self.model_path1 = os.path.join(base_project_root, "output_ep8/few_shot/...")
  self.model_path2 = os.path.join(base_project_root, "output_ep8/few_shot/...")
  ```
- 调整了数据集路径：
  ```python
  cfg.DATASET.ROOT = os.path.join(project_root, "data")
  ```
- 调整了key_image路径：
  ```python
  key_image_path = os.path.join(current_dir, "database", "key_image.png")
  ```

#### 2.3 添加API接口
- `/health` - 健康检查
- `/detect_watermark` - 检测水印（预留接口）
- `/start_dataset_test` - 开始数据集测试
- `/dataset_test_status` - 获取测试状态
- `/dataset_results` - 获取测试结果

#### 2.4 添加线程支持
- 测试在后台线程中运行，不阻塞主服务
- 支持状态查询和结果获取

#### 2.5 错误处理
- 添加了完善的异常处理
- 导入失败时提供友好的错误信息

## 使用方法

### 1. 启动服务
```bash
cd 四个服务/
python main_app.py
```

### 2. 访问API

#### 健康检查
```bash
curl http://localhost:3001/api/dataset_detection/health
```

#### 开始数据集测试
```bash
curl -X POST http://localhost:3001/api/dataset_detection/start_dataset_test
```

#### 查询测试状态
```bash
curl http://localhost:3001/api/dataset_detection/dataset_test_status
```

#### 获取测试结果
```bash
curl http://localhost:3001/api/dataset_detection/dataset_results
```

### 3. 测试脚本
运行测试脚本验证修改是否正确：
```bash
python test_dataset.py
```

## 配置说明

### 路径配置
- `project_root`: `/root/project/yun/FAP/lm-watermarking-main/`
- `base_project_root`: `/root/project/yun/FAP/`
- `current_dir`: 当前脚本所在目录

### 模型配置
- 配置文件：`configs/trainers/FAP/vit_b32_ep10_batch4_2ctx_notransform.yaml`
- 数据集配置：`configs/datasets/caltech101.yaml`
- 模型文件路径已内置，无需手动指定

## 注意事项

1. **依赖模块**：确保所有依赖模块都在正确的路径下
2. **GPU支持**：需要CUDA环境支持
3. **内存要求**：数据集测试需要较大的内存和GPU内存
4. **文件权限**：确保对database目录有写入权限

## 错误排查

### 常见问题

1. **导入失败**
   - 检查Python路径配置
   - 确保所有依赖模块存在

2. **模型加载失败**
   - 检查模型文件路径
   - 确保模型文件存在且完整

3. **路径错误**
   - 检查key_image.png是否存在
   - 确保database目录存在

4. **GPU内存不足**
   - 减少batch size
   - 使用更小的模型

## 与原版本的对比

| 特性 | 原版本 | 新版本 |
|------|--------|--------|
| 运行方式 | 命令行 | Flask API |
| 参数传递 | 命令行参数 | 内置配置 |
| 并发支持 | 无 | 线程支持 |
| 状态查询 | 无 | 实时状态 |
| 错误处理 | 基础 | 完善 |
| 集成性 | 独立 | 与main_app集成 |

## 测试结果

测试结果包含以下指标：
- 常规文本提示下常规样本准确率
- 常规文本提示下水印样本准确率  
- 水印文本提示下常规样本准确率
- 水印文本提示下水印样本准确率
- 伪造文本提示下常规样本准确率
- 伪造文本提示下水印样本准确率 
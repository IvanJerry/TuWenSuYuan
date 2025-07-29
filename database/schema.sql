-- TuWenSuYuan PostgreSQL 数据库结构
-- 创建数据库
CREATE DATABASE tuwensuyuan WITH ENCODING 'UTF8' LC_COLLATE='en_US.UTF-8' LC_CTYPE='en_US.UTF-8';

-- 启用扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. 用户表
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user'
);

-- 2. 密钥图片表
CREATE TABLE watermark_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    key_path VARCHAR(500) NOT NULL,
    description TEXT,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. 模型表
CREATE TABLE models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    model_type VARCHAR(20) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    accuracy FLOAT,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. 检测记录表
CREATE TABLE detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    model_id UUID REFERENCES models(id),
    key_id UUID REFERENCES watermark_keys(id),
    image_path VARCHAR(500),
    watermark_detected BOOLEAN,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. 配置表
CREATE TABLE user_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    key_name VARCHAR(100) NOT NULL,
    value TEXT NOT NULL,
    UNIQUE(user_id, key_name)
);

-- 创建索引
CREATE INDEX idx_detections_user_id ON detections(user_id);
CREATE INDEX idx_detections_created_at ON detections(created_at);

-- 插入默认数据
INSERT INTO users (username, email, password_hash, role) VALUES 
('admin', 'admin@tuwensuyuan.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/HS.iK8O', 'admin');

INSERT INTO watermark_keys (name, key_path, description, created_by) VALUES 
('default_key', '/static/image/key.png', '默认水印密钥图片', (SELECT id FROM users WHERE username = 'admin'));

INSERT INTO models (name, model_type, file_path, accuracy, created_by) VALUES 
('FAP_vit_b32', 'FAP', '/root/project/yun/FAP/output_ep8/few_shot/adv_term-cos_eps-1_alpha-1_train_iter-2_test_iter-100_lambda1-1.5/FAP_vit_b32_ep10_batch4_2ctx_notransform/caltech101_shot16/seed0/MultiModalPromptLearner/model.pth.tar-8', 82.88, (SELECT id FROM users WHERE username = 'admin'));

INSERT INTO user_configs (user_id, key_name, value) VALUES 
((SELECT id FROM users WHERE username = 'admin'), 'watermark_threshold', '0.5'),
((SELECT id FROM users WHERE username = 'admin'), 'max_upload_size', '10485760'); 
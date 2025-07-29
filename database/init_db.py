#!/usr/bin/env python3
"""
TuWenSuYuan PostgreSQL 数据库初始化脚本
"""

import psycopg2
import os
import sys
from pathlib import Path

def check_dependencies():
    """检查依赖"""
    try:
        import psycopg2
        print("✅ psycopg2 已安装")
    except ImportError:
        print("❌ 缺少 psycopg2 依赖")
        print("请运行: pip install psycopg2-binary")
        return False
    return True

def read_schema_file():
    """读取数据库结构文件"""
    schema_file = Path(__file__).parent / "schema.sql"
    if not schema_file.exists():
        print(f"❌ 找不到数据库结构文件: {schema_file}")
        return None
    
    with open(schema_file, 'r', encoding='utf-8') as f:
        return f.read()

def get_db_config():
    """获取数据库配置"""
    # 从环境变量获取配置
    config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'tuwensuyuan'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', ''),
    }
    
    print("📋 数据库配置:")
    print(f"  主机: {config['host']}")
    print(f"  端口: {config['port']}")
    print(f"  数据库: {config['database']}")
    print(f"  用户: {config['user']}")
    
    return config

def create_database(config):
    """创建数据库"""
    # 连接到默认数据库
    conn_config = config.copy()
    conn_config['database'] = 'postgres'  # 连接到默认数据库
    
    try:
        conn = psycopg2.connect(**conn_config)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # 检查数据库是否存在
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (config['database'],))
        exists = cursor.fetchone()
        
        if not exists:
            print(f"🔧 创建数据库: {config['database']}")
            cursor.execute(f"CREATE DATABASE {config['database']} WITH ENCODING 'UTF8'")
            print("✅ 数据库创建成功")
        else:
            print(f"✅ 数据库 {config['database']} 已存在")
        
        cursor.close()
        conn.close()
        
    except psycopg2.Error as e:
        print(f"❌ 创建数据库失败: {e}")
        return False
    
    return True

def init_database(config, schema_sql):
    """初始化数据库表结构"""
    try:
        # 连接到目标数据库
        conn = psycopg2.connect(**config)
        conn.autocommit = True
        cursor = conn.cursor()
        
        print("🔧 执行数据库初始化...")
        
        # 执行 SQL 脚本
        cursor.execute(schema_sql)
        
        print("✅ 数据库初始化成功")
        
        # 验证表是否创建成功
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        tables = cursor.fetchall()
        print("📋 已创建的表:")
        for table in tables:
            print(f"  - {table[0]}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except psycopg2.Error as e:
        print(f"❌ 数据库初始化失败: {e}")
        return False

def test_connection(config):
    """测试数据库连接"""
    try:
        conn = psycopg2.connect(**config)
        cursor = conn.cursor()
        
        # 测试查询
        cursor.execute("SELECT version()")
        version = cursor.fetchone()
        print(f"✅ 数据库连接成功")
        print(f"📋 PostgreSQL 版本: {version[0]}")
        
        cursor.close()
        conn.close()
        return True
        
    except psycopg2.Error as e:
        print(f"❌ 数据库连接失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 TuWenSuYuan 数据库初始化")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 读取数据库结构
    schema_sql = read_schema_file()
    if not schema_sql:
        sys.exit(1)
    
    # 获取数据库配置
    config = get_db_config()
    
    # 测试连接
    print("\n🔍 测试数据库连接...")
    if not test_connection(config):
        print("💡 请检查数据库配置:")
        print("   1. 确保 PostgreSQL 服务正在运行")
        print("   2. 检查环境变量: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD")
        print("   3. 或者手动设置数据库连接参数")
        sys.exit(1)
    
    # 创建数据库
    print("\n🔧 创建数据库...")
    if not create_database(config):
        sys.exit(1)
    
    # 初始化数据库
    print("\n🔧 初始化数据库表结构...")
    if not init_database(config, schema_sql):
        sys.exit(1)
    
    print("\n🎉 数据库初始化完成!")
    print("💡 下一步:")
    print("   1. 启动后端服务: cd backend && python app.py")
    print("   2. 访问前端页面: frontend/watermark.html")

if __name__ == "__main__":
    main() 
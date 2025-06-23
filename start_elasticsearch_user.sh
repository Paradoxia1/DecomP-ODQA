#!/bin/bash

# Elasticsearch 启动脚本 (使用非 root 用户)

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 创建 esuser 用户（如果不存在）
if ! id "esuser" &>/dev/null; then
    echo_info "创建 esuser 用户..."
    useradd -m esuser
fi

# 查找 Elasticsearch 目录
ES_DIR=""
for dir in /root/autodl-tmp/elasticsearch-*; do
    if [ -d "$dir" ]; then
        ES_DIR="$dir"
        break
    fi
done

if [ -z "$ES_DIR" ]; then
    echo_error "未找到 Elasticsearch 安装目录"
    exit 1
fi

echo_info "找到 Elasticsearch 目录: $ES_DIR"

# 检查版本
VERSION=$(basename "$ES_DIR" | sed 's/elasticsearch-//')
echo_info "Elasticsearch 版本: $VERSION"

# 将 Elasticsearch 移动到 esuser 的家目录
ES_USER_DIR="/home/esuser/$(basename "$ES_DIR")"
echo_info "将 Elasticsearch 复制到: $ES_USER_DIR"

if [ ! -d "$ES_USER_DIR" ]; then
    cp -r "$ES_DIR" "$ES_USER_DIR"
fi

# 确保 esuser 拥有目录权限
chown -R esuser:esuser "$ES_USER_DIR"

echo_info "以用户 esuser 启动 Elasticsearch..."

# 切换到 esuser 用户并启动 Elasticsearch
su - esuser -c "
export ES_JAVA_OPTS='-Xms1g -Xmx2g'
cd '$ES_USER_DIR'
./bin/elasticsearch"

#!/bin/bash

# FastChat 启动脚本
# 用于启动 Qwen3-8B 模型的 FastChat 服务

set -e  # 遇到错误立即退出

# 配置参数
MODEL_PATH="/root/autodl-tmp/DecomP-ODQA/RAG/Qwen3-8B"
CONTROLLER_PORT=21001
WORKER_PORT=21002
API_PORT=8001

# 日志目录
LOG_DIR="$HOME/.cache/fastchat_logs"
PID_DIR="$HOME/.cache/fastchat_pids"

# 创建必要的目录
mkdir -p "$LOG_DIR"
mkdir -p "$PID_DIR"

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

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo_error "模型路径不存在: $MODEL_PATH"
    exit 1
fi

echo_info "模型路径: $MODEL_PATH"

# 停止现有服务的函数
stop_services() {
    echo_info "停止现有的 FastChat 服务..."
    
    # 停止 API 服务器
    if [ -f "$PID_DIR/api_server.pid" ]; then
        PID=$(cat "$PID_DIR/api_server.pid")
        if kill -0 "$PID" 2>/dev/null; then
            echo_info "停止 API 服务器 (PID: $PID)"
            kill -TERM "$PID" 2>/dev/null || true
            sleep 2
            kill -KILL "$PID" 2>/dev/null || true
        fi
        rm -f "$PID_DIR/api_server.pid"
    fi
    
    # 停止模型工作器
    if [ -f "$PID_DIR/model_worker.pid" ]; then
        PID=$(cat "$PID_DIR/model_worker.pid")
        if kill -0 "$PID" 2>/dev/null; then
            echo_info "停止模型工作器 (PID: $PID)"
            kill -TERM "$PID" 2>/dev/null || true
            sleep 2
            kill -KILL "$PID" 2>/dev/null || true
        fi
        rm -f "$PID_DIR/model_worker.pid"
    fi
    
    # 停止控制器
    if [ -f "$PID_DIR/controller.pid" ]; then
        PID=$(cat "$PID_DIR/controller.pid")
        if kill -0 "$PID" 2>/dev/null; then
            echo_info "停止控制器 (PID: $PID)"
            kill -TERM "$PID" 2>/dev/null || true
            sleep 2
            kill -KILL "$PID" 2>/dev/null || true
        fi
        rm -f "$PID_DIR/controller.pid"
    fi
    
    # 清理端口
    echo_info "清理端口..."
    fuser -k $CONTROLLER_PORT/tcp 2>/dev/null || true
    fuser -k $WORKER_PORT/tcp 2>/dev/null || true
    fuser -k $API_PORT/tcp 2>/dev/null || true
    
    sleep 2
}

# 检查端口是否被占用
check_port() {
    local port=$1
    if netstat -tuln | grep -q ":$port "; then
        echo_warn "端口 $port 已被占用"
        return 1
    fi
    return 0
}

# 等待服务启动
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    echo_info "等待 $service_name 启动..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s --connect-timeout 1 "$url" >/dev/null 2>&1; then
            echo_info "✓ $service_name 启动成功"
            return 0
        fi
        echo "尝试 $attempt/$max_attempts..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo_error "✗ $service_name 启动失败"
    return 1
}

# 启动控制器
start_controller() {
    echo_info "启动 FastChat 控制器..."
    
    python -m fastchat.serve.controller \
        --host 127.0.0.1 \
        --port $CONTROLLER_PORT \
        > "$LOG_DIR/controller.log" 2>&1 &
    
    local pid=$!
    echo $pid > "$PID_DIR/controller.pid"
    echo_info "控制器已启动 (PID: $pid)"
    
    # 等待控制器启动
    if ! wait_for_service "http://127.0.0.1:$CONTROLLER_PORT" "控制器"; then
        return 1
    fi
}

# 启动模型工作器
start_model_worker() {
    echo_info "启动模型工作器..."
    
    python -m fastchat.serve.model_worker \
        --host 127.0.0.1 \
        --port $WORKER_PORT \
        --model-names "Qwen3-8B" \
        --model-path "$MODEL_PATH" \
        --controller "http://127.0.0.1:$CONTROLLER_PORT" \
        > "$LOG_DIR/model_worker.log" 2>&1 &
    
    local pid=$!
    echo $pid > "$PID_DIR/model_worker.pid"
    echo_info "模型工作器已启动 (PID: $pid)"
    
    # 等待模型加载（更长时间）
    echo_info "等待模型加载完成..."
    sleep 15
    
    # 检查模型是否注册成功
    local max_attempts=60
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -X POST "http://127.0.0.1:$CONTROLLER_PORT/list_models" | grep -q "models"; then
            echo_info "✓ 模型工作器注册成功"
            return 0
        fi
        echo "等待模型注册... $attempt/$max_attempts"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo_error "✗ 模型工作器注册失败"
    return 1
}

# 启动 API 服务器
start_api_server() {
    echo_info "启动 OpenAI API 服务器..."
    
    python -m fastchat.serve.openai_api_server \
        --controller "http://127.0.0.1:$CONTROLLER_PORT" \
        --port $API_PORT \
        --host 127.0.0.1 \
        > "$LOG_DIR/api_server.log" 2>&1 &
    
    local pid=$!
    echo $pid > "$PID_DIR/api_server.pid"
    echo_info "API 服务器已启动 (PID: $pid)"
    
    # 等待 API 服务器启动
    if ! wait_for_service "http://127.0.0.1:$API_PORT/v1/models" "API 服务器"; then
        return 1
    fi
}

# 显示服务状态
show_status() {
    echo ""
    echo "=========================="
    echo "FastChat 服务状态"
    echo "=========================="
    echo "控制器:      http://127.0.0.1:$CONTROLLER_PORT"
    echo "模型工作器:  http://127.0.0.1:$WORKER_PORT"
    echo "API 服务器:  http://127.0.0.1:$API_PORT"
    echo ""
    echo "测试命令:"
    echo "curl http://127.0.0.1:$API_PORT/v1/models"
    echo ""
    echo "停止服务:"
    echo "./stop_fastchat.sh"
    echo "=========================="
}

# 主函数
main() {
    echo "FastChat 启动脚本"
    echo "模型: Qwen3-8B"
    echo "===================="
    
    # 停止现有服务
    stop_services
    
    # 检查端口
    if ! check_port $CONTROLLER_PORT || ! check_port $WORKER_PORT || ! check_port $API_PORT; then
        echo_error "端口被占用，请先停止相关服务"
        exit 1
    fi
    
    # 按顺序启动服务
    if start_controller && start_model_worker && start_api_server; then
        echo_info "✓ 所有服务启动成功!"
        show_status
    else
        echo_error "✗ 服务启动失败"
        stop_services
        exit 1
    fi
}

# 捕获中断信号
trap 'echo_info "收到中断信号，停止服务..."; stop_services; exit 0' INT TERM

# 执行主函数
main "$@"

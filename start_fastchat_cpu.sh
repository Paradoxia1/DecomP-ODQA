#!/bin/bash

# FastChat 启动脚本 (CPU 版本)
# 用于在 CPU 上启动 Qwen3-8B 模型的 FastChat 服务

set -e  # 遇到错误立即退出

# 配置参数
MODEL_PATH="/home/ubuntu/Test-DecomP-ODQA/RAG/Qwen3-8B"
CONTROLLER_PORT=21001
WORKER_PORT=21002
API_PORT=8000

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
echo_info "模型大小: $(du -sh "$MODEL_PATH" | cut -f1)"

# 检查内存
TOTAL_MEM=$(free -m | awk 'NR==2{printf "%.1f", $2/1024}')
AVAILABLE_MEM=$(free -m | awk 'NR==2{printf "%.1f", $7/1024}')
echo_info "系统内存: ${TOTAL_MEM}GB 总计, ${AVAILABLE_MEM}GB 可用"

if (( $(echo "$AVAILABLE_MEM < 16" | bc -l) )); then
    echo_warn "可用内存 ${AVAILABLE_MEM}GB 可能不足以加载 16GB 模型"
    echo_warn "建议使用量化版本或更小的模型"
fi

# 停止现有服务的函数
stop_services() {
    echo_info "停止现有的 FastChat 服务..."
    pkill -f "fastchat.serve" 2>/dev/null || true
    sleep 3
}

# 等待服务启动
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    echo_info "等待 $service_name 启动..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s --connect-timeout 2 "$url" >/dev/null 2>&1; then
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

# 启动控制器 (如果没运行)
start_controller() {
    if curl -s "http://localhost:$CONTROLLER_PORT" >/dev/null 2>&1; then
        echo_info "控制器已在运行"
        return 0
    fi
    
    echo_info "启动 FastChat 控制器..."
    
    python -m fastchat.serve.controller \
        --port $CONTROLLER_PORT \
        > controller.log 2>&1 &
    
    local pid=$!
    echo_info "控制器已启动 (PID: $pid)"
    
    # 等待控制器启动
    if ! wait_for_service "http://localhost:$CONTROLLER_PORT" "控制器"; then
        return 1
    fi
}

# 启动模型工作器 (CPU 模式)
start_model_worker_cpu() {
    echo_info "启动模型工作器 (CPU 模式)..."
    echo_warn "注意: 在 CPU 上加载 16GB 模型需要很长时间..."
    
    # 设置环境变量强制使用 CPU
    export CUDA_VISIBLE_DEVICES=""
    export OMP_NUM_THREADS=8  # 限制 CPU 线程数
    
    python -m fastchat.serve.model_worker \
        --model-path "$MODEL_PATH" \
        --device cpu \
        --controller "http://localhost:$CONTROLLER_PORT" \
        --port $WORKER_PORT \
        --worker "http://localhost:$WORKER_PORT" \
        --load-8bit \
        > model_worker_cpu.log 2>&1 &
    
    local pid=$!
    echo_info "模型工作器已启动 (PID: $pid)"
    
    # 等待模型加载（CPU 需要更长时间）
    echo_info "等待模型加载完成 (可能需要 5-10 分钟)..."
    
    # 显示加载进度
    local max_attempts=300  # 10分钟超时
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        # 检查进程是否还在运行
        if ! kill -0 $pid 2>/dev/null; then
            echo_error "模型工作器进程已退出"
            echo "最后 20 行日志:"
            tail -20 model_worker_cpu.log
            return 1
        fi
        
        # 检查模型是否注册成功
        if curl -s -X POST "http://localhost:$CONTROLLER_PORT/list_models" | grep -q "models"; then
            echo_info "✓ 模型工作器注册成功"
            return 0
        fi
        
        # 每30秒显示一次进度
        if [ $((attempt % 15)) -eq 0 ]; then
            echo "加载进度: $((attempt * 2))/$((max_attempts * 2)) 秒"
            echo "最新日志:"
            tail -3 model_worker_cpu.log | sed 's/^/  /'
        fi
        
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo_error "✗ 模型工作器注册超时"
    echo "完整日志:"
    cat model_worker_cpu.log
    return 1
}

# 启动 API 服务器 (如果没运行)
start_api_server() {
    if curl -s "http://localhost:$API_PORT/v1/models" >/dev/null 2>&1; then
        echo_info "API 服务器已在运行"
        return 0
    fi
    
    echo_info "启动 OpenAI API 服务器..."
    
    python -m fastchat.serve.openai_api_server \
        --controller "http://localhost:$CONTROLLER_PORT" \
        --port $API_PORT \
        --host localhost \
        > openai_api_server.log 2>&1 &
    
    local pid=$!
    echo_info "API 服务器已启动 (PID: $pid)"
    
    # 等待 API 服务器启动
    if ! wait_for_service "http://localhost:$API_PORT/v1/models" "API 服务器"; then
        return 1
    fi
}

# 显示服务状态
show_status() {
    echo ""
    echo "=========================="
    echo "FastChat 服务状态 (CPU 模式)"
    echo "=========================="
    echo "控制器:      http://localhost:$CONTROLLER_PORT"
    echo "模型工作器:  http://localhost:$WORKER_PORT (CPU)"
    echo "API 服务器:  http://localhost:$API_PORT"
    echo ""
    echo "可用模型:"
    curl -s "http://localhost:$API_PORT/v1/models" | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for model in data.get('data', []):
        print(f\"  - {model['id']}\")
except:
    print('  (无法获取模型列表)')
"
    echo ""
    echo "测试命令:"
    echo "python test_fastchat_api.py"
    echo ""
    echo "停止服务:"
    echo "./stop_fastchat.sh"
    echo "=========================="
}

# 检查依赖
check_dependencies() {
    echo_info "检查依赖..."
    
    if ! command -v python >/dev/null 2>&1; then
        echo_error "Python 未安装"
        exit 1
    fi
    
    if ! python -c "import fastchat" 2>/dev/null; then
        echo_error "FastChat 未安装，请运行: pip install fschat"
        exit 1
    fi
    
    if ! command -v bc >/dev/null 2>&1; then
        echo_info "安装 bc 计算器..."
        sudo apt-get update && sudo apt-get install -y bc
    fi
    
    echo_info "依赖检查完成"
}

# 主函数
main() {
    echo "FastChat CPU 启动脚本"
    echo "模型: Qwen3-8B (CPU 模式)"
    echo "===================="
    
    # 检查依赖
    check_dependencies
    
    # 按顺序启动服务
    if start_controller && start_model_worker_cpu && start_api_server; then
        echo_info "✓ 所有服务启动成功!"
        show_status
    else
        echo_error "✗ 服务启动失败"
        echo_error "请检查日志文件:"
        echo "  - controller.log"
        echo "  - model_worker_cpu.log"
        echo "  - openai_api_server.log"
        exit 1
    fi
}

# 捕获中断信号
trap 'echo_info "收到中断信号，停止服务..."; stop_services; exit 0' INT TERM

# 执行主函数
main "$@"

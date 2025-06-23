#!/bin/bash

# FastChat 停止脚本

set -e

# 配置参数
CONTROLLER_PORT=21001
WORKER_PORT=21002
API_PORT=8001

# 目录
PID_DIR="$HOME/.cache/fastchat_pids"

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

# 停止单个服务
stop_service() {
    local service_name=$1
    local pid_file="$PID_DIR/${service_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo_info "停止 $service_name (PID: $pid)"
            
            # 尝试优雅停止
            if kill -TERM "$pid" 2>/dev/null; then
                # 等待进程结束
                local count=0
                while kill -0 "$pid" 2>/dev/null && [ $count -lt 10 ]; do
                    sleep 1
                    count=$((count + 1))
                done
                
                # 如果还没结束，强制停止
                if kill -0 "$pid" 2>/dev/null; then
                    echo_warn "强制停止 $service_name"
                    kill -KILL "$pid" 2>/dev/null || true
                fi
            fi
            
            echo_info "✓ $service_name 已停止"
        else
            echo_warn "$service_name 进程不存在"
        fi
        
        # 删除 PID 文件
        rm -f "$pid_file"
    else
        echo_warn "$service_name PID 文件不存在"
    fi
}

# 清理端口
cleanup_ports() {
    echo_info "清理端口..."
    
    # 强制终止占用端口的进程
    for port in $API_PORT $WORKER_PORT $CONTROLLER_PORT; do
        if netstat -tuln | grep -q ":$port "; then
            echo_info "清理端口 $port"
            fuser -k $port/tcp 2>/dev/null || true
        fi
    done
    
    sleep 2
}

# 主函数
main() {
    echo "停止 FastChat 服务..."
    echo "===================="
    
    # 按相反顺序停止服务
    stop_service "api_server"
    stop_service "model_worker"
    stop_service "controller"
    
    # 清理端口
    cleanup_ports
    
    echo_info "✓ 所有 FastChat 服务已停止"
}

# 执行主函数
main "$@"

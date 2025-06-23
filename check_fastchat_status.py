#!/usr/bin/env python3
"""
FastChat 服务状态检查脚本
检查所有组件是否正常运行
"""

import requests
import json
import subprocess
import time
import socket
from pathlib import Path

def check_port(host, port):
    """检查端口是否开放"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def check_processes():
    """检查 FastChat 相关进程"""
    print("🔍 检查 FastChat 进程...")
    
    processes = {
        "Controller": "fastchat.serve.controller",
        "Model Worker": "fastchat.serve.model_worker", 
        "API Server": "fastchat.serve.openai_api_server"
    }
    
    for name, process_name in processes.items():
        try:
            result = subprocess.run(
                ["pgrep", "-f", process_name], 
                capture_output=True, 
                text=True
            )
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                print(f"✓ {name}: 运行中 (PID: {', '.join(pids)})")
            else:
                print(f"✗ {name}: 未运行")
        except Exception as e:
            print(f"✗ {name}: 检查失败 ({e})")

def check_ports():
    """检查端口状态"""
    print("\n🌐 检查端口状态...")
    
    ports = {
        "Controller": ("127.0.0.1", 21001),
        "Model Worker": ("127.0.0.1", 21002),
        "API Server": ("127.0.0.1", 8001)
    }
    
    for name, (host, port) in ports.items():
        if check_port(host, port):
            print(f"✓ {name}: 端口 {port} 开放")
        else:
            print(f"✗ {name}: 端口 {port} 未开放")

def check_controller():
    """检查 Controller 状态"""
    print("\n🎮 检查 Controller...")
    
    try:
        # 检查基本连通性
        response = requests.get("http://127.0.0.1:21001/test", timeout=5)
        if response.status_code == 200:
            print("✓ Controller 响应正常")
        else:
            print(f"! Controller 响应异常: {response.status_code}")
    except Exception as e:
        print(f"✗ Controller 连接失败: {e}")
        return False
    
    try:
        # 检查注册的模型
        response = requests.post("http://127.0.0.1:21001/list_models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"✓ 注册的模型: {models.get('models', [])}")
            return len(models.get('models', [])) > 0
        else:
            print(f"✗ 获取模型列表失败: {response.status_code}")
    except Exception as e:
        print(f"✗ 获取模型列表失败: {e}")
    
    return False

def check_api_server():
    """检查 API Server 状态"""
    print("\n🚀 检查 API Server...")
    
    try:
        # 检查模型列表
        response = requests.get("http://localhost:8001/v1/models", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            print("✓ API Server 响应正常")
            print(f"完整响应: {json.dumps(models_data, indent=2, ensure_ascii=False)}")
            
            if models_data.get('data'):
                print(f"✓ 可用模型数量: {len(models_data['data'])}")
                for model in models_data['data']:
                    print(f"  - {model['id']}")
                return models_data['data']
            else:
                print("✗ 没有可用模型")
                return []
        else:
            print(f"✗ API Server 响应错误: {response.status_code}")
            print(f"响应内容: {response.text}")
    except Exception as e:
        print(f"✗ API Server 连接失败: {e}")
    
    return []

def test_model_inference(models):
    """测试模型推理"""
    if not models:
        print("\n❌ 没有可用模型，跳过推理测试")
        return False
        
    print("\n🧠 测试模型推理...")
    
    model_name = models[0]['id']
    print(f"使用模型: {model_name}")
    
    # 测试简单的文本生成
    test_cases = [
        {
            "prompt": "你好",
            "max_tokens": 10,
            "temperature": 0.1
        },
        {
            "prompt": "1+1=",
            "max_tokens": 5,
            "temperature": 0.0
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: '{test_case['prompt']}'")
        try:
            payload = {
                "model": model_name,
                "prompt": test_case["prompt"],
                "max_tokens": test_case["max_tokens"],
                "temperature": test_case["temperature"],
                "top_p": 1.0
            }
            
            response = requests.post(
                "http://localhost:8001/v1/completions",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('choices') and len(result['choices']) > 0:
                    generated_text = result['choices'][0]['text']
                    print(f"✓ 生成成功: '{generated_text.strip()}'")
                else:
                    print("✗ 响应格式异常")
                    print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
                    return False
            else:
                print(f"✗ 请求失败: {response.status_code}")
                print(f"错误: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ 推理测试失败: {e}")
            return False
    
    print("✅ 模型推理测试通过!")
    return True

def check_logs():
    """检查日志文件"""
    print("\n📋 检查日志文件...")
    
    log_files = [
        "controller.log",
        "worker.log", 
        "api.log"
    ]
    
    for log_file in log_files:
        if Path(log_file).exists():
            print(f"✓ 发现日志文件: {log_file}")
            # 显示最后几行
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    if lines:
                        print(f"  最后几行:")
                        for line in lines[-3:]:
                            print(f"    {line.strip()}")
            except Exception as e:
                print(f"  读取失败: {e}")
        else:
            print(f"✗ 日志文件不存在: {log_file}")

def check_model_files():
    """检查模型文件"""
    print("\n📁 检查模型文件...")
    
    model_path = Path("/root/autodl-tmp/DecomP-ODQA/RAG/Qwen3-8B")
    
    if model_path.exists():
        print(f"✓ 模型目录存在: {model_path}")
        
        # 检查关键文件
        key_files = [
            "config.json",
            "tokenizer.json", 
            "tokenizer_config.json"
        ]
        
        for file_name in key_files:
            file_path = model_path / file_name
            if file_path.exists():
                print(f"  ✓ {file_name}")
            else:
                print(f"  ✗ {file_name}")
        
        # 检查模型权重文件
        weight_files = list(model_path.glob("*.bin")) + list(model_path.glob("*.safetensors"))
        if weight_files:
            print(f"  ✓ 发现 {len(weight_files)} 个权重文件")
        else:
            print("  ✗ 没有发现权重文件")
            
    else:
        print(f"✗ 模型目录不存在: {model_path}")

def main():
    """主检查函数"""
    print("=" * 60)
    print("🔍 FastChat 服务状态检查")
    print("=" * 60)
    
    # 基础检查
    check_processes()
    check_ports()
    check_model_files()
    
    # 服务功能检查
    controller_ok = check_controller()
    models = check_api_server()
    
    # 推理测试
    if controller_ok and models:
        inference_ok = test_model_inference(models)
    else:
        inference_ok = False
    
    # 日志检查
    check_logs()
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 检查总结")
    print("=" * 60)
    
    if controller_ok and models and inference_ok:
        print("🎉 所有检查通过! FastChat 服务运行正常")
        print("\n可以开始使用 FastChat API:")
        print("  - API 地址: http://localhost:8001")
        print(f"  - 可用模型: {[m['id'] for m in models]}")
    else:
        print("❌ 检查发现问题，请检查上面的错误信息")
        print("\n故障排除建议:")
        print("1. 重新启动服务: ./start_fastchat.sh")
        print("2. 查看详细日志: cat controller.log worker.log api.log")
        print("3. 检查模型文件完整性")

if __name__ == "__main__":
    main()

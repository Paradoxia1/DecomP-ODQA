#!/usr/bin/env python3
"""
FastChat API 测试脚本
"""

import requests
import json
import time

def test_fastchat_api():
    """测试 FastChat API"""
    api_base = "http://localhost:8001"
    
    print("FastChat API 测试")
    print("=" * 40)
    
    # 1. 测试服务是否可达
    print("1. 检查服务连接...")
    try:
        response = requests.get(f"{api_base}/v1/models", timeout=5)
        if response.status_code == 200:
            print("✓ 服务连接正常")
            models = response.json()
            models_list = models.get('data', [])
            print(f"可用模型数量: {len(models_list)}")
            for model in models_list:
                print(f"  - {model.get('id', 'unknown')}")
            if not models_list:
                print("  ⚠️ 警告：没有可用模型！")
                print("     这通常意味着模型工作器没有成功注册")
                print("     请检查模型工作器日志")
                return False
        else:
            print(f"✗ 服务响应错误: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        return False
    
    # 2. 测试文本生成
    print("\n2. 测试文本生成...")
    try:
        # 获取第一个可用模型
        models_response = requests.get(f"{api_base}/v1/models", timeout=5)
        models_data = models_response.json()
        if models_data.get('data'):
            model_name = models_data['data'][0]['id']
            print(f"从服务器获取的模型名: {model_name}")
        else:
            print("✗ 无法获取可用模型列表")
            return False
        
        # 发送生成请求
        payload = {
            "model": model_name,
            "prompt": "Hello, how are you?",
            "max_tokens": 50,
            "temperature": 0.7,
            "top_p": 1.0,
            "stop": ["\n"]
        }
        
        print(f"使用模型: {model_name}")
        print(f"提示词: {payload['prompt']}")
        
        response = requests.post(
            f"{api_base}/v1/completions",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('choices'):
                generated_text = result['choices'][0]['text']
                print("✓ 文本生成成功")
                print(f"生成结果: {generated_text.strip()}")
            else:
                print("✗ 响应格式异常")
                print(f"响应: {result}")
        else:
            print(f"✗ 生成失败: {response.status_code}")
            error_info = response.text
            print(f"错误信息: {error_info}")
            
            # 如果是模型名称错误，尝试重新获取模型列表
            if "model" in error_info.lower() and response.status_code == 400:
                print("\n尝试重新获取正确的模型名称...")
                try:
                    models_response = requests.get(f"{api_base}/v1/models", timeout=5)
                    if models_response.status_code == 200:
                        models_data = models_response.json()
                        available_models = [m['id'] for m in models_data.get('data', [])]
                        print(f"服务器上的实际模型: {available_models}")
                except:
                    pass
            return False
            
    except Exception as e:
        print(f"✗ 生成请求失败: {e}")
        return False
    
    # 3. 测试聊天接口（如果支持）
    print("\n3. 测试聊天接口...")
    try:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{api_base}/v1/chat/completions",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('choices'):
                message = result['choices'][0]['message']['content']
                print("✓ 聊天接口正常")
                print(f"回答: {message.strip()}")
            else:
                print("✗ 聊天响应格式异常")
        else:
            print(f"! 聊天接口可能不支持: {response.status_code}")
            
    except Exception as e:
        print(f"! 聊天接口测试失败: {e}")
    
    print("\n" + "=" * 40)
    print("✓ API 测试完成")
    return True

def check_controller_status():
    """检查控制器状态和注册的工作器"""
    controller_base = "http://localhost:21001"
    
    print("\n检查控制器状态...")
    print("=" * 40)
    
    # 1. 检查控制器是否可达
    print("1. 控制器连接性测试...")
    try:
        response = requests.get(controller_base, timeout=5)
        print(f"✓ 控制器响应: {response.status_code}")
    except Exception as e:
        print(f"✗ 控制器连接失败: {e}")
        return False
    
    # 2. 检查注册的工作器
    print("\n2. 检查注册的模型工作器...")
    try:
        response = requests.post(f"{controller_base}/list_models", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 控制器响应正常")
            print(f"注册信息: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"✗ 控制器响应错误: {response.status_code}")
            print(f"响应内容: {response.text}")
    except Exception as e:
        print(f"✗ 控制器查询失败: {e}")
    
    return False

def main():
    """主函数"""
    print("等待服务启动...")
    time.sleep(2)
    
    # 首先检查控制器状态
    check_controller_status()
    
    # 重试机制
    max_retries = 3
    for i in range(max_retries):
        print(f"\n尝试 {i+1}/{max_retries}")
        if test_fastchat_api():
            break
        if i < max_retries - 1:
            print("等待 10 秒后重试...")
            time.sleep(10)
    else:
        print("\n✗ 所有测试尝试都失败了")
        print("请检查:")
        print("1. FastChat 服务是否正常启动")
        print("2. 模型是否正确加载")
        print("3. 端口 8001 是否可访问")
        print("4. 模型工作器是否成功注册到控制器")
        print("\n建议检查日志文件:")
        print("- controller.log 或 ~/.cache/fastchat_logs/controller.log")
        print("- model_worker_cpu.log 或 ~/.cache/fastchat_logs/model_worker.log")
        print("- openai_api_server.log 或 ~/.cache/fastchat_logs/api_server.log")
    
    # 检查控制器状态
    check_controller_status()

if __name__ == "__main__":
    main()

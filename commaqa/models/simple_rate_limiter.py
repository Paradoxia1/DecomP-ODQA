import time
import threading

# 全局变量
_last_request_time = 0
_request_lock = threading.Lock()
_min_interval = 4.5  # 每个请求之间最少间隔4.5秒（12~15请求/分钟）

def wait_for_rate_limit():
    """简单的全局速率限制 - 确保请求间隔至少5秒"""
    global _last_request_time
    
    with _request_lock:
        now = time.time()
        time_since_last = now - _last_request_time
        
        if time_since_last < _min_interval:
            wait_time = _min_interval - time_since_last
            print(f"Rate limiting: waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        
        _last_request_time = time.time()
        print("Request permitted")
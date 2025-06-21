import logging
import time
import os
import json
import requests
from functools import lru_cache
from typing import List, Tuple, Optional

from diskcache import Cache
from commaqa.inference.prompt_reader import fit_prompt_into_given_limit


logger = logging.getLogger(__name__)


cache = Cache(os.path.expanduser("~/.cache/fastchat_calls"))


@cache.memoize()
def cached_fastchat_call(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    stop: List[str],
    n: int,
    stream: bool = False,
    api_base: str = "http://localhost:8000/v1"
):
    """缓存的 FastChat API 调用"""
    return fastchat_completion(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        n=n,
        stream=stream,
        api_base=api_base
    )


def fastchat_completion(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    stop: List[str],
    n: int,
    stream: bool = False,
    api_base: str = "http://localhost:8000/v1"
):
    """调用 FastChat API"""
    url = f"{api_base}/completions"
    
    headers = {
        "Content-Type": "application/json",
    }
    
    data = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stop": stop,
        "n": n,
        "stream": stream,
        "logprobs": 1,  # 请求 logprobs 用于计算分数
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    return response.json()


def fastchat_call(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    stop: List[str],
    n: int,
    stream: bool = False,
    api_base: str = "http://localhost:8000/v1"
):
    """FastChat API 调用封装，支持缓存"""
    function = cached_fastchat_call if temperature == 0 else fastchat_completion
    return function(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        n=n,
        stream=stream,
        api_base=api_base
    )


@lru_cache(maxsize=1)
def get_tokenizer(model_name: str = "gpt2"):
    """获取分词器，用于估算 token 数量"""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except:
        # 如果指定模型不可用，回退到 GPT2
        from transformers import GPT2Tokenizer
        return GPT2Tokenizer.from_pretrained("gpt2")


class FastChatGenerator:
    def __init__(
        self,
        model: str = "qwen3:8b",  # 默认模型名称
        api_base: str = "http://localhost:8000/v1",  # FastChat API 地址
        temperature: float = 0,
        max_tokens: int = 300,
        top_p: float = 1,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        stop: List[str] = None,
        retry_after_n_seconds: Optional[int] = None,
        n: int = 1,
        remove_method: str = "first",
        model_tokens_limit: int = 4096,  # 模型的 token 限制
        tokenizer_model_name: str = "gpt2",  # 用于 token 计算的分词器
    ):
        self.model = model
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop if stop is not None else ["\n"]
        self.retry_after_n_seconds = retry_after_n_seconds
        self.n = n
        self.remove_method = remove_method
        self.model_tokens_limit = model_tokens_limit
        self.tokenizer_model_name = tokenizer_model_name

        # 测试连接
        self._test_connection()

    def _test_connection(self):
        """测试 FastChat 服务连接"""
        try:
            models_url = f"{self.api_base}/models"
            response = requests.get(models_url, timeout=5)
            response.raise_for_status()
            models_data = response.json()
            
            available_models = [model["id"] for model in models_data.get("data", [])]
            logger.info(f"Available models: {available_models}")
            
            if self.model not in available_models:
                logger.warning(f"Model '{self.model}' not found in available models. Available: {available_models}")
                if available_models:
                    self.model = available_models[0]
                    logger.info(f"Using first available model: {self.model}")
                    
        except Exception as e:
            logger.error(f"Failed to connect to FastChat service at {self.api_base}: {e}")
            raise ConnectionError(f"Cannot connect to FastChat service: {e}")

    def generate_text_sequence(self, prompt: str) -> List[Tuple[str, float]]:
        """
        生成文本序列
        :param prompt: 输入提示
        :return: 返回 (文本, 分数) 元组列表，分数越低越好
        """
        # 清理提示末尾的空白字符
        prompt = prompt.rstrip()

        # 调整提示长度以适应模型限制
        prompt = fit_prompt_into_given_limit(
            original_prompt=prompt,
            model_length_limit=self.model_tokens_limit,
            estimated_generation_length=self.max_tokens,
            demonstration_delimiter="\n\n\n",
            shuffle=False,
            remove_method=self.remove_method,
            tokenizer_model_name=self.tokenizer_model_name,
            last_is_test_example=True,
        )

        arguments = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "n": self.n,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop,
            "api_base": self.api_base,
        }

        success = False
        for index in range(10):  # 减少重试次数
            try:
                response = fastchat_call(**arguments)
                success = True
                break
            except Exception as exception:
                success = False

                # 如果是 token 限制问题，尝试减少 max_tokens
                if "maximum context length" in str(exception).lower() or "token" in str(exception).lower():
                    tokenizer = get_tokenizer(self.tokenizer_model_name)
                    prompt_num_tokens = len(tokenizer.encode(prompt))
                    
                    if prompt_num_tokens + arguments["max_tokens"] > self.model_tokens_limit > prompt_num_tokens:
                        last_used_max_tokens = arguments["max_tokens"]
                        updated_max_tokens = self.model_tokens_limit - prompt_num_tokens - 10  # 留一些余量
                        arguments["max_tokens"] = max(1, updated_max_tokens)
                        
                        if last_used_max_tokens == arguments["max_tokens"]:
                            break
                            
                        logger.warning(
                            f"WARNING: (Round {index}) Decreasing max_tokens from "
                            f"{last_used_max_tokens} to {arguments['max_tokens']} and retrying."
                        )
                        continue

                if self.retry_after_n_seconds is None:
                    logger.error(f"FastChat API call failed: {exception}")
                    raise exception

                logger.warning(f"Encountered exception: {exception.__class__.__name__}")
                logger.warning(f"Will retry in {self.retry_after_n_seconds}s.")
                time.sleep(self.retry_after_n_seconds)

        if not success:
            raise Exception("Could not complete FastChat API call")

        # 处理响应
        output_seq_score = []

        for index, choice in enumerate(response.get("choices", [])):
            text = choice.get("text", "")
            
            # 尝试从 logprobs 计算分数
            if "logprobs" in choice and choice["logprobs"]:
                logprobs_data = choice["logprobs"]
                if "token_logprobs" in logprobs_data and logprobs_data["token_logprobs"]:
                    token_logprobs = logprobs_data["token_logprobs"]
                    tokens = logprobs_data.get("tokens", [])
                    
                    probs = []
                    for i, (prob, tok) in enumerate(zip(token_logprobs, tokens)):
                        if prob is not None:  # 有些 token 可能没有 logprob
                            if tok not in self.stop and tok != "<|endoftext|>":
                                probs.append(prob)
                            else:
                                probs.append(prob)
                                break
                    
                    score = -sum(probs) / len(probs) if len(probs) > 0 else 100.0
                    output_seq_score.append((text, score))
                else:
                    # 如果没有有效的 logprobs，使用索引作为分数
                    output_seq_score.append((text, float(index)))
            else:
                # 没有 logprobs 信息，使用索引作为分数
                output_seq_score.append((text, float(index)))

        # 按分数排序，分数越低越好
        return sorted(output_seq_score, key=lambda x: x[1])

    def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        try:
            models_url = f"{self.api_base}/models"
            response = requests.get(models_url, timeout=5)
            response.raise_for_status()
            models_data = response.json()
            return [model["id"] for model in models_data.get("data", [])]
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []

    def set_model(self, model_name: str):
        """设置使用的模型"""
        available_models = self.get_available_models()
        if model_name in available_models:
            self.model = model_name
            logger.info(f"Model set to: {model_name}")
        else:
            logger.error(f"Model '{model_name}' not available. Available models: {available_models}")
            raise ValueError(f"Model '{model_name}' not available")

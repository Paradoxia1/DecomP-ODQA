import logging
import os
import json
import requests
import time
import random
from functools import lru_cache
from diskcache import Cache
from commaqa.models.simple_rate_limiter import wait_for_rate_limit
from commaqa.inference.prompt_reader import fit_prompt_into_given_limit

logger = logging.getLogger(__name__)

cache = Cache(os.path.expanduser("~/.cache/geminicalls"))


@cache.memoize()
def cached_gemini_call(prompt, model, temperature, max_tokens, top_p, stop, n, frequency_penalty, presence_penalty):
    """Cached version of Gemini API call for temperature=0"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    # 在发送请求前等待速率限制
    # wait_for_rate_limit()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    # Build the request payload according to Gemini API format
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "topP": top_p,
            "candidateCount": n  # Use n instead of candidate_count for consistency
        }
    }
    
    # Add stop sequences if provided
    if stop and len(stop) > 0:
        payload["generationConfig"]["stopSequences"] = stop
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"Gemini API call failed with status {response.status_code}: {response.text}")
    
    return response.json()


def gemini_call(prompt, model, temperature, max_tokens, top_p, stop, n, frequency_penalty, presence_penalty):
    """Make a Gemini API call, with caching for temperature=0"""    
    if temperature == 0:
        return cached_gemini_call(prompt, model, temperature, max_tokens, top_p, stop, n, frequency_penalty, presence_penalty)
    else:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        # 在发送请求前等待速率限制
        # wait_for_rate_limit()

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": top_p,
                "candidateCount": n
            }
        }
        
        if stop and len(stop) > 0:
            payload["generationConfig"]["stopSequences"] = stop
        
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Gemini API call failed with status {response.status_code}: {response.text}")
        
        return response.json()


@lru_cache(maxsize=1)
def get_gemini_tokenizer():
    """Get a tokenizer for Gemini (using GPT2 tokenizer as approximation)"""
    from transformers import GPT2Tokenizer
    return GPT2Tokenizer.from_pretrained("gpt2")


class GeminiGenerator:
    def __init__(
        self,
        engine="gemini-2.0-flash-lite",  # Use correct Gemini model name
        temperature=0,
        max_tokens=300,  # Match GPT3Generator default
        top_p=1,
        frequency_penalty=0,  # Add for consistency (not used by Gemini)
        presence_penalty=0,   # Add for consistency (not used by Gemini)
        stop=["\n"],
        retry_after_n_seconds=None,
        n=1,
        best_of=1,  # Add for consistency (not used by Gemini)
        logprobs=0,  # Add for consistency (not used by Gemini)
        remove_method="first",  # Add for consistency
    ):
        self.engine = engine
        self.logprobs = logprobs
        self.n = n
        self.best_of = best_of
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stop = stop
        self.temperature = temperature
        self.retry_after_n_seconds = retry_after_n_seconds
        self.remove_method = remove_method
        
        # Validate Gemini model name
        valid_gemini_models = [
            "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro", 
            "gemini-1.0-pro", "gemini-pro", "gemini-pro-vision"
        ]
        if not any(model in engine for model in valid_gemini_models):
            raise Exception(f"Invalid Gemini model: {engine}. Must contain one of: {valid_gemini_models}")
        
        # Set model token limits (approximate values for Gemini)
        if "gemini-2.0" in engine:
            self.model_tokens_limit = 32000  # Gemini 2.0 has higher limits
        elif "gemini-1.5" in engine:
            self.model_tokens_limit = 32000
        else:
            self.model_tokens_limit = 8000  # Default conservative limit
        
        # Validate API key is available
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    def generate_text_sequence(self, prompt):
        """
        :param prompt: input text prompt
        :return: returns a sequence of tuples (string, score) where lower score is better
        """
        # Gemini API doesn't handle trailing whitespace well
        prompt = prompt.rstrip()
        
        # Apply prompt length limiting (same as GPT3Generator)
        prompt = fit_prompt_into_given_limit(
            original_prompt=prompt,
            model_length_limit=self.model_tokens_limit,
            estimated_generation_length=self.max_tokens,
            demonstration_delimiter="\n\n\n",
            shuffle=False,
            remove_method=self.remove_method,
            tokenizer_model_name="gpt2",  # Use GPT2 tokenizer as approximation
            last_is_test_example=True,
        )
        
        arguments = {
            "model": self.engine,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "n": self.n,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop,
        }
        
        success = False
        for index in range(500):  # Same retry logic as GPT3Generator
            try:
                response = gemini_call(**arguments)
                success = True
                break
            except Exception as exception:
                success = False
                
                # Handle token limit exceeded (similar to GPT3Generator)
                tokenizer = get_gemini_tokenizer()
                prompt_num_tokens = len(tokenizer.tokenize(prompt))
                if prompt_num_tokens + arguments["max_tokens"] > self.model_tokens_limit > prompt_num_tokens:
                    last_used_max_tokens = arguments["max_tokens"]
                    updated_max_tokens = self.model_tokens_limit - prompt_num_tokens
                    arguments["max_tokens"] = updated_max_tokens
                    if last_used_max_tokens == updated_max_tokens:
                        break
                    print(
                        f"WARNING: (Round {index}) Decreasing max_tokens from "
                        f"{last_used_max_tokens} to {updated_max_tokens} and retrying."
                    )
                    continue
                
                if self.retry_after_n_seconds is None:
                    import traceback
                    print(traceback.format_exc())
                    exit()
                
                print(f"Encountered exception of class: {exception.__class__}")
                print(str(exception))
                print(f"Potentially reached Gemini rate limit. Will try again in {self.retry_after_n_seconds}s.")
                time.sleep(self.retry_after_n_seconds)
                pass
        
        if not success:
            raise Exception("Could not complete Gemini call")
        
        output_seq_score = []
        
        # Parse Gemini API response format
        if "candidates" in response:
            for index, candidate in enumerate(response["candidates"]):
                if "content" in candidate and "parts" in candidate["content"]:
                    # Extract text from the candidate
                    text = ""
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            text += part["text"]
                    
                    # Handle stop sequences manually since Gemini may not always respect them
                    if self.stop:
                        for stop_str in self.stop:
                            if stop_str in text:
                                text = text[:text.index(stop_str)]
                                break
                    
                    # Score calculation (similar to GPT3Generator logic)
                    # Since Gemini doesn't provide logprobs, use index-based scoring
                    score = index
                    
                    # Apply additional scoring based on safety and quality
                    if "safetyRatings" in candidate:
                        safety_score = 0
                        for rating in candidate["safetyRatings"]:
                            if rating.get("probability", "NEGLIGIBLE") != "NEGLIGIBLE":
                                safety_score += 1
                        score += safety_score
                    
                    # Check if the response was blocked
                    finish_reason = candidate.get("finishReason", "")
                    if "SAFETY" in finish_reason:
                        score += 100  # Heavily penalize blocked responses
                        
                    output_seq_score.append((text, score))
                
                else:
                    # Handle cases where content was blocked or filtered
                    logger.warning(f"Candidate {index} has no content, likely filtered")
                    output_seq_score.append(("", 100))  # High penalty score
        
        else:
            # Handle error cases
            if "error" in response:
                error_msg = response["error"].get("message", "Unknown error")
                logger.error(f"Gemini API error: {error_msg}")
                raise Exception(f"Gemini API error: {error_msg}")
            else:
                logger.error(f"Unexpected response format: {response}")
                output_seq_score.append(("", 100))
        
        # If no valid responses, return empty with high penalty
        if not output_seq_score:
            output_seq_score.append(("", 100))
        
        # Ensure sorted output (lower score is better, same as GPT3Generator)
        return sorted(output_seq_score, key=lambda x: x[1])

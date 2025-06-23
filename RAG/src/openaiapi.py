import json
import logging
import os
from dotenv import load_dotenv
from openai import OpenAI
from openai import AzureOpenAI


# Initialize logger
logger = logging.getLogger(__name__)

# # Load environment variables from .env file
# load_dotenv()
# # 清除代理环境变量
# proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']
# for var in proxy_vars:
#     if var in os.environ:
#         del os.environ[var]
        
class OpenaiLLM:
    """
    A class to interact with OpenAI or Azure OpenAI services.
    Defaults to Azure OpenAI if no specific mode is provided.
    """

    def __init__(self,base_url = "http://0.0.0.0:8001/v1"):
        """
        Initialize the OpenaiLLM class with the provided arguments.

        Args:
            args (argparse.Namespace): Arguments containing necessary configurations.
        """
        

        # Initialize the client based on the mode (Azure OpenAI or Native OpenAI)
        
        self.client = OpenAI(api_key="none", base_url = base_url)
        self.MODEL_NAME = "Qwen3-8B"
        logger.info("Initialized OpenAI LLM in Native OpenAI mode.")
    

        logger.info("OpenaiLLM initialization completed.")

    def run(self, system_msg, prompt):
        """
        Run the LLM with the provided system message and user question.

        Args:
            system_msg (str): The system message to set the context.
            question (str): The user's question or prompt.

        Returns:
            str: The response from the LLM.
        """
        res = None
        try:
            assert len(prompt) > 0, "Question cannot be empty."

            # Prepare the messages for the LLM
            if system_msg:
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [
                    {"role": "user", "content": prompt}
                ]

            # Call the LLM API
            completion = self.client.chat.completions.create(
                model=self.MODEL_NAME,  # Updated to "gpt-4o"
                messages=messages,
                # max_tokens=8192,  #本地运行加入这个最大token限制，思维链计算在内，小了容易爆出
                # temperature=getattr(self.args, 'temperature', 0.6), #Qwen3使用默认配置在推理下最好
                max_tokens=8192,
                temperature=0.6,
                top_p=0.95,
                extra_body={
                    "top_k": 20, 
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            )

            # Calculate the cost of the API call
            # cost = calc_price(model=self.MODEL_NAME, usage=completion.usage)  # Updated to "gpt-4o"

            # Extract the response from the completion
            res = completion.choices[0].message.content
            # reasoning = completion.choices[0].message.reasoning_content
            # Log the interaction details
            interaction_details = {
                "system_msg": system_msg,
                "prompt": prompt,
                "llm_response": res,
                # "llm_reasoning": reasoning,
                # 'cost': cost,
                "success": True,
                "model": "openai_"+self.MODEL_NAME
            }
            logger.info(f"OpenaiLLM Interaction | {json.dumps(interaction_details, ensure_ascii=False)}")
        except Exception as e:
            # Log the interaction details
            interaction_details = {
                "system_msg": system_msg,
                "prompt": prompt,
                "llm_response": res,
                "success": False,
                "msg": e,
                "model": "openai_"+self.MODEL_NAME
            }
            logger.info(f"OpenaiLLM Interaction | {json.dumps(interaction_details, ensure_ascii=False)}")
        return res


if __name__ == '__main__':

    # from args_tool import get_args
    # args = get_args()
    llm = OpenaiLLM()

    print(llm.run(
        system_msg="",
        prompt="hi, 1+1=?"
    ))

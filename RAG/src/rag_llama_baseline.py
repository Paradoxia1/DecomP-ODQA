import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

import torch
# import vllm
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from openaiapi import OpenaiLLM
import json
from multiprocessing.pool import ThreadPool
# os.environ['RAY_TMPDIR'] = "/data_8T/gsk/huawei/tmp"
# import ray

"""1. blingfire 库
blingfire 是一个快速、轻量级的自然语言处理（NLP）工具库，它提供了多种文本处理功能，像文本分词、句子分割等。
该库以速度快著称，在处理大规模文本数据时表现出色。

2. text_to_sentences_and_offsets 函数
text_to_sentences_and_offsets 函数的主要功能是将输入的文本分割成句子，同时返回每个句子在原始文本中的起始和结束偏移量。
借助偏移量，能够精准定位每个句子在原始文本里的位置。
"""
######################################################################################################
######################################################################################################
###
### IMPORTANT !!!
### Before submitting, please follow the instructions in the docs below to download and check in :
### the model weighs.
###
###  https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/download_baseline_model_weights.md
###
### And please pay special attention to the comments that start with "TUNE THIS VARIABLE"
###                        as they depend on your model and the available GPU resources.
###
### DISCLAIMER: This baseline has NOT been tuned for performance
###             or efficiency, and is provided as is for demonstration.
######################################################################################################


# Load the environment variable that specifies the URL of the MockAPI. This URL is essential
# for accessing the correct API endpoint in Task 2 and Task 3. The value of this environment variable
# may vary across different evaluation settings, emphasizing the importance of dynamically obtaining
# the API URL to ensure accurate endpoint communication.

# Please refer to https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/crag-mock-api
# for more information on the MockAPI.
#
# **Note**: This environment variable will not be available for Task 1 evaluations.


CRAG_MOCK_API_URL = os.getenv("CRAG_MOCK_API_URL", "http://localhost:8001")


#### CONFIG PARAMETERS ---

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 20
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1000
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4000

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 8 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 4 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.95 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 128 # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

#### CONFIG PARAMETERS END---
POROCESS_NUM = 4

class ChunkExtractor:

    def _extract_chunks(self, interaction_id, html_source):
        """
        @ray.remote 装饰器的主要功能是把一个函数或类方法转换为 Ray 的远程任务（对于函数）或者远程 actor（对于类）。
        这意味着被装饰的函数或方法能在 Ray 集群的不同节点上异步执行，从而实现并行计算和分布式处理。
        从给定的 HTML 源代码里提取并返回文本块（chunks）。
        Extracts and returns chunks from given HTML source.
        
        Note: This function is for demonstration purposes only.
        We are treating an independent sentence as a chunk here,
        but you could choose to chunk your text more cleverly than this.

        Parameters:
            interaction_id (str): Interaction ID that this HTML source belongs to.
            html_source (str): HTML content from which to extract text.

        Returns:
            Tuple[str, List[str]]: A tuple containing the interaction ID and a list of sentences extracted from the HTML content.
        """
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_source, "lxml")
        text = soup.get_text(" ", strip=True)  # Use space as a separator, strip whitespaces

        if not text:
            # Return a list with empty string when no text is extracted
            return interaction_id, [""]

        # Extract offsets of sentences from the text
        _, offsets = text_to_sentences_and_offsets(text)

        # Initialize a list to store sentences
        chunks = []

        # Iterate through the list of offsets and extract sentences
        for start, end in offsets:
            # Extract the sentence and limit its length
            sentence = text[start:end][:MAX_CONTEXT_SENTENCE_LENGTH]
            chunks.append(sentence)

        return interaction_id, chunks

    def extract_chunks(self, batch_interaction_ids, batch_search_results):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.

        Parameters:
            batch_interaction_ids (List[str]): List of interaction IDs.
            batch_search_results (List[List[Dict]]): List of search results batches, each containing HTML text.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        # Setup parallel chunk extraction using ray remote
        if POROCESS_NUM != -1:
            pool = ThreadPool(POROCESS_NUM)
            response_refs = [pool.apply_async(self._extract_chunks, args=(batch_interaction_ids[idx], html_text["page_result"]))
                             for idx, search_results in enumerate(batch_search_results)
                             for html_text in search_results]
            pool.close()
            pool.join()
            response_refs_results = [response_ref.get() for response_ref in response_refs]

        # Wait until all sentence extractions are complete
        # and collect chunks for every interaction_id separately
            chunk_dictionary = defaultdict(list)
            for interaction_id, _chunks  in response_refs_results:
                chunk_dictionary[interaction_id].extend(_chunks)
        else:
            chunk_dictionary = defaultdict(list)
            for idx, search_results in enumerate(batch_search_results):
                interaction_id = batch_interaction_ids[idx]
                for html_text in search_results:
                    _interaction_id, _chunks = self._extract_chunks(interaction_id, html_text["page_result"])
                    chunk_dictionary[_interaction_id].extend(_chunks)
                    
        # Flatten chunks and keep a map of corresponding interaction_ids
        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)

        return chunks, chunk_interaction_ids

    def _flatten_chunks(self, chunk_dictionary):
        """
        Flattens the chunk dictionary into separate lists for chunks and their corresponding interaction IDs.

        Parameters:
            chunk_dictionary (defaultdict): Dictionary with interaction IDs as keys and lists of chunks as values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        # Convert to numpy arrays for convenient slicing/masking operations later
        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids

class RAGModel:
    """
    An example RAGModel for the KDDCup 2024 Meta CRAG Challenge
    which includes all the key components of a RAG lifecycle.
    """
    def __init__(self):
        self.llm = OpenaiLLM(base_url="http://0.0.0.0:8001/v1")

        # Load a sentence transformer model optimized for sentence embeddings, using CUDA if available.
        self.sentence_model = SentenceTransformer(
            "all-MiniLM-L6-v2",  # 使用模型名称，会自动从Hugging Face下载
            device=torch.device(
                "cuda:3" if torch.cuda.is_available() else "cpu"
            ),
        )
        self.chunk_extractor = ChunkExtractor()

    def initialize_models(self):
        # Initialize Meta Llama 3 - 8B Instruct Model
        self.model_name = "/root/autodl-tmp/DecomP-ODQA/RAG/Qwen3-8B"

        if not os.path.exists(self.model_name):
            raise Exception(
                f"""
            The evaluators expect the model weights to be checked into the repository,
            but we could not find the model weights at {self.model_name}
            
            Please follow the instructions in the docs below to download and check in the model weights.
            
            https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md
            """
            )

        # Initialize the model with vllm
        # self.llm = vllm.LLM(
        #     "/data_8T/gsk/Qwen3-8B",
        #     served_model_name="Qwen3-32B",
        #     worker_use_ray=True,
        #     tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE, 
        #     gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
        #     trust_remote_code=True,

        # )
        # self.tokenizer = self.llm.get_tokenizer()
        
        self.llm = OpenaiLLM(base_url="http://0.0.0.0:8001/v1")

        # Load a sentence transformer model optimized for sentence embeddings, using CUDA if available.
        self.sentence_model = SentenceTransformer(
            "all-MiniLM-L6-v2",  # 使用模型名称，会自动从Hugging Face下载
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )

    def calculate_embeddings(self, sentences):
        """
        Compute normalized embeddings for a list of sentences using a sentence encoding model.

        This function leverages multiprocessing to encode the sentences, which can enhance the
        processing speed on multi-core machines.

        Args:
            sentences (List[str]): A list of sentences for which embeddings are to be computed.

        Returns:
            np.ndarray: An array of normalized embeddings for the given sentences.

        """
        embeddings = self.sentence_model.encode(
            sentences=sentences,
            normalize_embeddings=True,
            batch_size=SENTENTENCE_TRANSFORMER_BATCH_SIZE,
        )
        # Note: There is an opportunity to parallelize the embedding generation across 4 GPUs
        #       but sentence_model.encode_multi_process seems to interefere with Ray
        #       on the evaluation servers. 
        #       todo: this can also be done in a Ray native approach.
        #       
        return embeddings

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.
        
        The evaluation timeouts linearly scale with the batch size. 
            i.e.: time out for the `batch_generate_answer` call = batch_size * per_sample_timeout 
        

        Returns:
            int: The batch size, an integer between 1 and 16. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE  
        return self.batch_size

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates answers for a batch of queries using associated (pre-cached) search results and query times.

        Parameters:
            batch (Dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'interaction_id;  (List[str]): List of interaction_ids for the associated queries
                - 'query' (List[str]): List of user queries.
                - 'search_results' (List[List[Dict]]): List of search result lists, each corresponding
                                                      to a query. Please refer to the following link for
                                                      more details about the individual search objects:
                                                      https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md#search-results-detail
                - 'query_time' (List[str]): List of timestamps (represented as a string), each corresponding to when a query was made.

        Returns:
            List[str]: A list of plain text responses for each query in the batch. Each response is limited to 75 tokens.
            If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        # Chunk all search results using ChunkExtractor
        chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
            batch_interaction_ids, batch_search_results
        )

        # Calculate all chunk embeddings
        chunk_embeddings = self.calculate_embeddings(chunks)

        # Calculate embeddings for queries
        query_embeddings = self.calculate_embeddings(queries)

        # Retrieve top matches for the whole batch
        batch_retrieval_results = []
        for _idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[_idx]
            query_time = query_times[_idx]
            query_embedding = query_embeddings[_idx]

            # Identify chunks that belong to this interaction_id
            relevant_chunks_mask = chunk_interaction_ids == interaction_id

            # Filter out the said chunks and corresponding embeddings
            relevant_chunks = chunks[relevant_chunks_mask]
            relevant_chunks_embeddings = chunk_embeddings[relevant_chunks_mask]

            # Calculate cosine similarity between query and chunk embeddings,
            cosine_scores = (relevant_chunks_embeddings * query_embedding).sum(1)

            # and retrieve top-N results.
            retrieval_results = relevant_chunks[
                (-cosine_scores).argsort()[:NUM_CONTEXT_SENTENCES]  #5个网页的所有信息综合，提取最高的NUM_CONTEXT_SENTENCES=20条
            ]
            
            # You might also choose to skip the steps above and 
            # use a vectorDB directly.
            batch_retrieval_results.append(retrieval_results)
            
        system_prompt, user_messages = self.format_prompts(queries, query_times, batch_retrieval_results)

        

        # 创建线程池，线程数量可根据实际情况调整
        if POROCESS_NUM != -1:
            pool = ThreadPool(processes=POROCESS_NUM)
            # 并行执行任务
            async_results = [pool.apply_async(self.llm.run, (system_prompt, user_message)) for user_message in user_messages] 
            #返回的不是最终结果，而是一个 AsyncResult 对象，可以把它想象成一张**“任务回执单”或“快递单号”**。results 这个列表最终会包含所有任务的“回执单”。
            pool.close()  #你不能再向 pool 添加任何新任务。但已经在工作或排队的助理会继续完成他们手头的任务。
            pool.join()  #: 这是一个阻塞（Blocking）操作。主程序（经理）会在这里停下来，一直等待，直到线程池中所有的任务（所有助理）都已经执行完毕。
            answers = [result.get() for result in async_results]

        else:
            answers = []
            for user_message in user_messages:
                answers.append(self.llm.run(system_prompt, user_message))
        return answers


    def format_prompts(self, queries, query_times, batch_retrieval_results=[]):
        """
        Formats queries, corresponding query_times and retrieval results using the chat_template of the model.
            
        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        - query_times (List[str]): A list of query_time strings corresponding to each query.
        - batch_retrieval_results (List[str])
        """        
        system_prompt = "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. There is no need to explain the reasoning behind your answers."
        user_message_list = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]

            user_message = ""
            references = ""
            
            if len(retrieval_results) > 0:
                references += "# References \n"
                # Format the top sentences as references in the model's prompt template.
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    references += f"- {snippet.strip()}\n"
            
            # references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
            # Limit the length of references to fit the model's input size.

            user_message += f"{references}\n------\n\n"
            user_message 
            user_message += f"Using only the references listed above, answer the following question: \n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"
            user_message_list.append(user_message)
            
        return system_prompt, user_message_list #这里的prompt可能过于简单，但是对于8B模型不知道如何，输出格式也不明确。


if __name__ == "__main__":
    # Create an instance of the RAGModel
    from load_data import RAGBenchmarkDataset
    from torch.utils.data import DataLoader
    
    
    data_path = "/root/autodl-tmp/DecomP-ODQA/RAG/crag_task_1_dev_v4_release.jsonl"
    dataset=RAGBenchmarkDataset(file_path=data_path)
    dataloader = DataLoader(
        batch_size=2,
        dataset=dataset,
        shuffle=False,
        collate_fn=RAGBenchmarkDataset.custom_collate_fn
    )
    rag_model = RAGModel()
    # chunk_model = ChunkExtractor()
    
    cur_batch = next(iter(dataloader))
    answer = rag_model.batch_generate_answer(cur_batch)
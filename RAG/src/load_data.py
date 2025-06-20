import json
import os
import pickle
from typing import List, Dict, Any
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader

# ===================================================================
# 1. 定义我们自己的 Dataset 类
# ===================================================================
class RAGBenchmarkDataset(Dataset):
    """
    一个用于加载 RAG 基准测试数据集的自定义 PyTorch Dataset。
    它从一个 JSON Lines (.jsonl) 文件中加载数据。
    """
    def __init__(self, file_path: str):
        """
        初始化 Dataset.
        
        Args:
            file_path (str): 数据集文件的路径 (应为 .jsonl 格式).
        """
        self.data = []
        pkl_file_path = file_path.replace('.jsonl', '.pkl')

        if os.path.exists(pkl_file_path):
            try:
                # 尝试从 .pkl 文件加载数据
                with open(pkl_file_path, 'rb') as f:
                    self.data = pickle.load(f)
                print(f"成功从 {pkl_file_path} 加载 {len(self.data)} 条数据")
                return
            except Exception as e:
                print(f"从 {pkl_file_path} 加载数据失败: {e}，尝试从原始 JSONL 文件加载")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # 解析文件中的每一行作为一个独立的 JSON 对象
                    self.data.append(json.loads(line))
            print(f"成功加载 {len(self.data)} 条数据，来源: {file_path}")

            # 将解析后的数据保存为 .pkl 文件
            with open(pkl_file_path, 'wb') as f:
                pickle.dump(self.data, f)
            print(f"已将数据保存到 {pkl_file_path}")
        except FileNotFoundError:
            print(f"错误: 数据文件未找到于 '{file_path}'")
            raise
        except json.JSONDecodeError:
            print(f"错误: 无法解析文件 '{file_path}' 中的 JSON 数据。请确保它是有效的 JSON Lines 格式。")
            raise

    def __len__(self) -> int:
        """返回数据集中的样本总数。"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        根据索引获取一个数据样本。
        我们只返回批次化所需的核心字段，以保持整洁。
        """
        sample = self.data[idx]
        return {
            'interaction_id': sample.get('interaction_id', ''),
            'query': sample.get('query', ''),
            'search_results': sample.get('search_results', []),
            'query_time': sample.get('query_time', '')
        }

# ===================================================================
# 2. 定义我们的自定义 collate_fn
# ===================================================================

# 你当前把 custom_collate_fn 定义为类方法（@classmethod），这会导致问题。类方法的第一个参数默认是类对象（通常命名为 cls），而 DataLoader 调用 collate_fn 时只会传入一个参数（即批次样本列表），这就使得 custom_collate_fn 接收到的参数数量和预期不符，从而抛出 TypeError 异常。

# 下面提供两种解决方案：

# 方案一：将 custom_collate_fn 改为静态方法
# 静态方法不需要接收类对象或实例对象作为第一个参数，正好符合 DataLoader 对 collate_fn 的调用要求。
    @staticmethod
    def custom_collate_fn(batch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        自定义的整理函数，将从 Dataset 中取出的样本列表
        转换成您指定的批次字典格式。

        Args:
            batch_samples (List[Dict[str, Any]]): 一个样本字典的列表。
                例如: [{'id': 'a', 'query': 'q1', ...}, {'id': 'b', 'query': 'q2', ...}]

        Returns:
            Dict[str, Any]: 一个整合后的批次字典。
                例如: {'id': ['a', 'b'], 'query': ['q1', 'q2'], ...}
        """
        # 使用 defaultdict 可以让代码更简洁
        collated_batch = defaultdict(list)
        
        for sample in batch_samples:
            collated_batch['interaction_id'].append(sample['interaction_id'])
            collated_batch['query'].append(sample['query'])
            collated_batch['search_results'].append(sample['search_results'])
            collated_batch['query_time'].append(sample['query_time'])
            
        return dict(collated_batch)

# ===================================================================
# 3. 创建一个虚拟数据集文件用于演示
# ===================================================================
# def create_dummy_data_file(filename="dummy_rag_data.jsonl", num_samples=5):
#     """创建一个假的 .jsonl 数据文件用于测试。"""
#     print(f"正在创建虚拟数据集文件: {filename}...")
#     with open(filename, 'w', encoding='utf-8') as f:
#         for i in range(num_samples):
#             data_point = {
#                 "interaction_id": f"id_{i:03}",
#                 "query_time": f"2024-06-12T10:00:0{i}Z",
#                 "domain": "sports",
#                 "question_type": "simple",
#                 "static_or_dynamic": "fast-changing",
#                 "query": f"Who won the match on day {i}?",
#                 "answer": f"Team A won on day {i}.",
#                 "alt_ans": [f"The winner was Team A on day {i}."],
#                 "split": 1,
#                 "search_results": [
#                     {
#                         "page_name": f"Match Report Day {i}",
#                         "page_url": f"http://example.com/report/{i}",
#                         "page_snippet": f"A detailed report of the match on day {i} where Team A emerged victorious.",
#                         "page_result": f"<html><body><h1>Match Day {i}</h1><p>Team A beat Team B.</p></body></html>",
#                         "page_last_modified": f"2024-06-12T09:00:0{i}Z"
#                     }
#                     for _ in range(3) # 每个query有3个搜索结果
#                 ]
#             }
#             f.write(json.dumps(data_point) + '\n')
#     print("虚拟文件创建完成。")

# ===================================================================
# 4. 主执行代码
# ===================================================================
if __name__ == "__main__":
    # DUMMY_FILE = "dummy_rag_data.jsonl"
    
    # # 创建虚拟数据文件
    # create_dummy_data_file(DUMMY_FILE)

    # (1) 创建数据集实例
    dataset = RAGBenchmarkDataset(file_path="/data_8T/gsk/huawei/crag_task_1_dev_v4_release.jsonl")

    # (2) 创建 DataLoader 实例，并传入我们的自定义 collate_fn
    #     我们设置 batch_size=3，所以每个批次会包含3个样本
    dataloader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=True,          # 在每个 epoch 开始时打乱数据
        collate_fn=custom_collate_fn
    )

    # (3) 迭代 DataLoader 来获取和检查批次
    print("\n" + "="*50)
    print("开始从 DataLoader 中迭代批次...")
    print("="*50)

    # next(iter(dataloader)) 只获取第一个批次用于演示
    first_batch = next(iter(dataloader))

    print("\n成功获取一个批次！批次格式如下:\n")
    
    # 使用 json.dumps 美化输出，使其更易读
    print(json.dumps(first_batch, indent=2))
    
    # 检查每个键的类型和长度
    print("\n" + "-"*50)
    print("验证批次中每个键的内容:")
    print("-"*50)
    for key, value in first_batch.items():
        print(f"Key: '{key}'")
        print(f"  - Type: {type(value)}")
        print(f"  - Length: {len(value)}")
        if isinstance(value, list) and len(value) > 0:
            print(f"  - Type of first element: {type(value[0])}")
    
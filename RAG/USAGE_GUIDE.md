# 如何使用example.jsonl进行预测

## 项目概述

这是一个基于Decomposed Prompting方法的开放域问答(ODQA)系统，用于解决复杂的多跳问题。系统采用分解-检索-推理的方法来回答需要多个推理步骤的问题。

## 数据格式要求

`example.jsonl`文件应该包含JSONL格式的数据，每行是一个JSON对象，包含以下字段：

### 必需字段
- `question_id`: 问题的唯一标识符
- `question_text`: 要回答的问题文本

### 可选字段
- `contexts`: 相关段落列表，每个段落包含：
  - `title`: 段落标题
  - `paragraph_text`: 段落内容
  - `is_supporting`: 是否为支撑段落（布尔值）
- `answers_objects`: 答案对象列表（用于评估）
- `level`: 问题难度级别
- `type`: 问题类型

### 示例数据格式
```json
{
  "question_id": "example_001", 
  "question_text": "What is the length of the river into which Pack Creek runs after it goes through the Spanish Valley?",
  "contexts": [
    {
      "title": "Spanish Valley",
      "paragraph_text": "Spanish Valley is a flat in Grand County, Utah, south of Moab...",
      "is_supporting": true
    }
  ],
  "answers_objects": [{"spans": ["1450 mi"]}]
}
```

## 预测流程

### 1. 准备环境
```bash
# 激活环境
conda activate decomp-odqa

# 启动必要的服务
./start_elasticsearch_user.sh  # 启动检索服务器
./start_fastchat.sh            # 启动LLM服务器（如果使用本地模型）
```

### 2. 配置选择
根据您的需求选择合适的配置文件：

- **数据集类型**: `hotpotqa`, `2wikimultihopqa`, `musique`
- **推理方法**: 
  - `decomp_context_cot_qa`: 带上下文的思维链分解
  - `decomp_context_direct_qa`: 直接问答分解
- **模型选择**:
  - `fastchat`: 本地FastChat模型
  - `codex`: OpenAI Codex
  - `flan_t5_*`: Google T5系列模型

### 3. 运行预测
使用以下命令进行预测：

```bash
python predict.py \
    --config base_configs/decomp_context_cot_qa_fastchat_hotpotqa.jsonnet \
    --input RAG/example.jsonl \
    --output predictions/my_prediction.json
```

### 4. 命令参数说明
- `config`: 配置文件路径，定义模型和推理策略
- `input`: 输入的JSONL文件路径
- `output`: 预测结果输出路径
- `--prediction-suffix`: 可选的预测目录后缀
- `--force`: 强制重新预测（覆盖已有结果）
- `--variable-replacements`: JSON格式的变量替换

### 5. 系统工作原理

#### 分解推理过程
1. **问题分解**: 将复杂问题分解为多个子问题
2. **信息检索**: 对每个子问题检索相关文档
3. **逐步推理**: 使用LLM逐步回答子问题
4. **答案合成**: 将子答案组合成最终答案

#### 核心组件
- **Decomposer**: 负责问题分解和推理控制
- **Retriever**: 基于Elasticsearch的文档检索
- **LLM Models**: 各种语言模型的统一接口
- **Execution Router**: 决定下一步执行哪个模型

## 配置文件详解

### 基础配置结构
```jsonnet
{
    "start_state": "decompose",           // 起始状态
    "end_state": "[EOQ]",                // 结束状态
    "models": {                          // 模型定义
        "decompose": { ... },            // 问题分解模型
        "retrieve": { ... },             // 检索模型
        "singlehop_titleqa": { ... },    // 单跳问答模型
        "multihop_titleqa": { ... }      // 多跳问答模型
    },
    "reader": {                          // 数据读取器配置
        "name": "multi_para_rc",
        "add_paras": false,
        "add_gold_paras": false
    }
}
```

### 模型类型说明
- **llmqadecomp**: LLM问答分解模型
- **retriever**: 文档检索模型
- **llmtitleqa**: 基于标题的问答模型
- **execute_router**: 执行路由器

## 输出结果

### 预测文件格式
```json
{
    "question_id_1": "predicted_answer_1",
    "question_id_2": "predicted_answer_2"
}
```

### 附加输出文件
- `*_chains.txt`: 推理链记录
- `*_time_taken.txt`: 执行时间
- `*_full_eval_path.txt`: 评估路径
- `config__*.jsonnet`: 配置文件备份

## 故障排除

### 常见问题
1. **服务器未启动**: 确保Elasticsearch和LLM服务器正在运行
2. **内存不足**: 减少检索文档数量或使用更小的模型
3. **API限制**: 如果使用OpenAI API，检查API密钥和限制

### 调试选项
```bash
# 启用详细输出
python predict.py config.jsonnet example.jsonl --debug

# 演示模式（交互式）
python -m commaqa.inference.configurable_inference \
    --config config.jsonnet \
    --demo
```

## 自定义使用

### 创建自定义配置
1. 复制现有配置文件
2. 修改模型参数和提示文件
3. 调整检索和推理策略

### 添加新的数据源
1. 确保数据格式符合要求
2. 更新Elasticsearch索引（如需要）
3. 调整数据读取器配置

## 性能优化

### 并行处理
```bash
# 使用多线程
python -m commaqa.inference.configurable_inference \
    --config config.jsonnet \
    --input example.jsonl \
    --output output.json \
    --threads 4
```

### 内存管理
- 定期保存中间结果（每5个样本）
- 支持断点续传
- 自动清理临时文件

## 注意事项

1. **数据质量**: 确保输入数据格式正确且完整
2. **计算资源**: 复杂问题可能需要较长处理时间
3. **模型选择**: 根据问题类型选择合适的模型配置
4. **服务依赖**: 确保所有必要的服务都正常运行

使用此系统时，建议先在小规模数据上测试，确认流程正常后再处理大批量数据。

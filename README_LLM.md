```markdown
# 推理模块使用说明（Turnstiles LLM Engine）

## 📖 简介
本模块提供了基于 Qwen2.5-3B-Instruct 的本地大模型推理接口。  
所有复杂的模型加载、GPU 调度、性能统计已封装，你只需调用 `generate()` 即可获得模型回答和详细的性能指标。

**适用硬件**：RTX 5060 笔记本 GPU（8GB 及以上显存）  
**核心文件**：`llm_engine.py`  
**环境要求**：Python 3.10、CUDA 12.8（驱动 ≥ 13.0 亦可兼容）

---

## 🚀 快速开始（10分钟上手）

### 1. 环境搭建
确保已安装 Anaconda，然后打开终端执行：

```bash
# 创建并激活环境
conda create -n turnstiles python=3.10 -y
conda activate turnstiles

# 安装依赖库（使用项目根目录下的 requirements.txt）
pip install -r requirements.txt
```

### 2. 下载模型
运行下载脚本（仅首次需要，约需 10~30 分钟）：
```bash
python download_model.py
```
模型将保存在 `./models/qwen/Qwen2.5-3B-Instruct/` 目录中。

### 3. 运行 Demo
```bash
python demo.py
```
根据提示输入问题（如“解释一下RAG”），你将看到模型回答和完整的性能统计。

---

## 📁 目录结构
```
turnstiles-llm/
├── models/                    # 模型权重（需自行下载，不提交到 Git）
│   └── qwen/Qwen2.5-3B-Instruct/
├── llm_engine.py              # 核心推理接口（对外暴露 generate 函数）
├── download_model.py          # 模型下载脚本
├── demo.py                    # 一键演示脚本
├── requirements.txt           # Python 依赖清单
└── README_LLM.md              # 本说明文件
```

---

## 🛠 接口说明

### `generate(prompt: str, max_new_tokens: int = 512) -> dict`

**功能**：根据用户输入的文本，生成模型回答并统计性能。

**参数**：
- `prompt` (str) : 用户输入的问题或指令。
- `max_new_tokens` (int, 可选) : 最大生成 token 数，默认 512。

**返回值**：
一个包含 `answer` 和 `latency` 两个字段的字典，格式如下：

```python
{
    "answer": "模型生成的文本回答",
    "latency": {
        "ttft": 0.1234,            # 首包响应时间（秒）
        "generation_time": 2.3456, # 纯生成耗时（秒）
        "total_time": 2.5678,      # 端到端总耗时（秒）
        "tokens_per_second": 45.67,# 生成速度（tokens/秒）
        "input_tokens": 12,        # 输入 token 数量
        "output_tokens": 100       # 输出 token 数量
    }
}
```

**使用示例**：
```python
from llm_engine import generate

result = generate("请简要介绍机器学习", max_new_tokens=256)
print(result["answer"])
print(result["latency"])
```

**错误处理**：
若推理过程出错（如显存不足），`answer` 会返回错误提示，`latency` 中所有值均为 0。

---

## ❓ 常见问题

**Q: 运行时报 CUDA out of memory 怎么办？**  
A: 把 `max_new_tokens` 调小（如 256）或关闭其他占用显存的程序。3B 模型在 8GB 显存下通常无压力。

**Q: 如何升级模型到 7B？**  
A: 修改 `download_model.py` 中的模型 ID 为 `qwen/Qwen2.5-7B-Instruct`，并更新 `llm_engine.py` 中的路径。注意 7B 模型需要 16GB 以上显存。

**Q: 首次加载模型很慢？**  
A: 正常，模型仅首次调用时加载（懒加载），后续调用会直接使用缓存。

---

## 📬 团队协作
- 模型权重文件 (`models/`) 请勿提交到 Git 仓库（已在 `.gitignore` 中忽略）。
- 队友拉取代码后只需执行“快速开始”中的三个步骤即可复现环境。

有任何问题请联系模块开发者：王晨曦。
```
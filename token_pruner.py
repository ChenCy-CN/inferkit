# token_pruner.py
"""
Token 剪枝器 —— 在输入进入模型之前，按重要性筛掉冗余 token

支持三种剪枝策略：
  1. head_tail: 保留前 N% 和后 M% 的 token
  2. similarity: 去掉和相邻 token 太相似的（去重）
  3. hybrid: 上面两种策略结合
"""

from transformers import AutoTokenizer
from loguru import logger
from typing import List, Optional
import torch
import re

# 全局分词器（加载一次就行）
_tokenizer = None
_model_path = "E:\code\Models\Qwen2.5-3B-Instruct\qwen\Qwen2___5-3B-Instruct"  # ← 改成你自己的路径


def _get_tokenizer():
    """懒加载分词器"""
    global _tokenizer
    if _tokenizer is None:
        logger.info("🔄 正在加载分词器用于剪枝...")
        _tokenizer = AutoTokenizer.from_pretrained(_model_path)
    return _tokenizer


def prune_by_head_tail(
        text: str,
        keep_ratio: float = 0.85,
        head_ratio: float = 0.7
) -> str:
    """
    头尾保留策略：保留前 (head_ratio * keep_ratio) 和后 ((1-head_ratio) * keep_ratio) 的 token

    直觉：开头通常是任务指令、结尾通常是约束条件，中间的信息相对次要。

    参数:
        text: 原始输入文本
        keep_ratio: 总共保留多少比例的 token（0~1），默认 0.85
        head_ratio: 保留的 token 中，头部占多少比例（0~1），默认 0.7

    返回:
        剪枝后的文本
    """
    tokenizer = _get_tokenizer()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    total = len(tokens)
    keep_count = int(total * keep_ratio)
    head_count = int(keep_count * head_ratio)
    tail_count = keep_count - head_count

    if keep_count >= total:
        logger.info(f"📌 Token 数 ({total}) 未超过阈值，无需剪枝")
        return text

    # 取头部和尾部 token
    pruned_tokens = tokens[:head_count] + tokens[-(tail_count):] if tail_count > 0 else tokens[:head_count]

    pruned_text = tokenizer.decode(pruned_tokens, skip_special_tokens=True)

    logger.info(f"✂️ 头尾剪枝: {total} tokens → {len(pruned_tokens)} tokens (保留 {keep_ratio * 100:.0f}%)")

    return pruned_text


def prune_by_similarity(
        text: str,
        similarity_threshold: float = 0.95,
        window_size: int = 3
) -> str:
    """
    相似度去重策略：去掉和上下文太相似的 token（避免重复信息）

    参数:
        text: 原始输入文本
        similarity_threshold: 相似度阈值（0~1），超过此值视为冗余，默认 0.95
        window_size: 比较窗口大小，默认 3

    返回:
        剪枝后的文本
    """
    tokenizer = _get_tokenizer()

    # 先把文本切成句子
    sentences = re.split(r'(?<=[。！？.!?])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1:
        logger.info("📌 只有 1 个句子，无需相似度剪枝")
        return text

    # 对每个句子编码，计算相邻句子间的余弦相似度
    kept_sentences = [sentences[0]]  # 第一个句子永远保留

    for i in range(1, len(sentences)):
        # 把当前句子和上一个保留的句子编码
        prev_tokens = tokenizer.encode(kept_sentences[-1], add_special_tokens=False)
        curr_tokens = tokenizer.encode(sentences[i], add_special_tokens=False)

        # 把 token 序列补到相同长度（后面补 0）
        max_len = max(len(prev_tokens), len(curr_tokens))
        prev_vec = prev_tokens + [0] * (max_len - len(prev_tokens))
        curr_vec = curr_tokens + [0] * (max_len - len(curr_tokens))

        # 算余弦相似度
        prev_tensor = torch.tensor(prev_vec, dtype=torch.float32)
        curr_tensor = torch.tensor(curr_vec, dtype=torch.float32)

        dot_product = torch.dot(prev_tensor, curr_tensor)
        norm_product = torch.norm(prev_tensor) * torch.norm(curr_tensor)

        if norm_product > 0:
            similarity = (dot_product / norm_product).item()
        else:
            similarity = 0

        # 相似度低于阈值 → 保留这个句子
        if similarity < similarity_threshold:
            kept_sentences.append(sentences[i])

    pruned_text = " ".join(kept_sentences)

    original_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    pruned_tokens = len(tokenizer.encode(pruned_text, add_special_tokens=False))

    logger.info(
        f"✂️ 相似度剪枝: {original_tokens} tokens → {pruned_tokens} tokens (去掉了 {len(sentences) - len(kept_sentences)} 个句子)")

    return pruned_text


def prune_hybrid(
        text: str,
        keep_ratio: float = 0.85,
        head_ratio: float = 0.7,
        similarity_threshold: float = 0.95
) -> str:
    """
    混合剪枝策略：先相似度去重，再头尾保留

    参数:
        text: 原始输入文本
        keep_ratio: 头尾剪枝的保留比例
        head_ratio: 头尾剪枝的头部比例
        similarity_threshold: 相似度剪枝的阈值

    返回:
        剪枝后的文本
    """
    logger.info("🔧 执行混合剪枝 (相似度去重 → 头尾保留)")

    # 第一步：相似度去重
    text_after_similarity = prune_by_similarity(text, similarity_threshold)

    # 第二步：头尾保留
    text_after_headtail = prune_by_head_tail(text_after_similarity, keep_ratio, head_ratio)

    return text_after_headtail


def prune_text(
        text: str,
        strategy: str = "hybrid",
        target_ratio: float = 0.85,
        **kwargs
) -> dict:
    """
    统一的剪枝入口函数

    参数:
        text: 原始输入文本
        strategy: 剪枝策略，可选 "head_tail"、"similarity"、"hybrid"
        target_ratio: 目标保留比例（0~1），默认 0.85
        **kwargs: 传给具体策略的额外参数

    返回:
        {
            "pruned_text": 剪枝后的文本,
            "original_tokens": 原始 token 数,
            "pruned_tokens": 剪枝后 token 数,
            "compression_ratio": 压缩比,
            "strategy": 使用的策略名
        }
    """
    tokenizer = _get_tokenizer()
    original_tokens = len(tokenizer.encode(text, add_special_tokens=False))

    # 选策略
    if strategy == "head_tail":
        pruned_text = prune_by_head_tail(text, keep_ratio=target_ratio, **kwargs)
    elif strategy == "similarity":
        pruned_text = prune_by_similarity(text, **kwargs)
    elif strategy == "hybrid":
        pruned_text = prune_hybrid(text, keep_ratio=target_ratio, **kwargs)
    else:
        raise ValueError(f"不支持的剪枝策略: {strategy}，可选: head_tail, similarity, hybrid")

    pruned_tokens = len(tokenizer.encode(pruned_text, add_special_tokens=False))
    compression_ratio = pruned_tokens / original_tokens if original_tokens > 0 else 1.0

    return {
        "pruned_text": pruned_text,
        "original_tokens": original_tokens,
        "pruned_tokens": pruned_tokens,
        "compression_ratio": round(compression_ratio, 4),
        "strategy": strategy
    }


# ═══════════════════════════════════════
# 快速自测
# ═══════════════════════════════════════
if __name__ == "__main__":
    # 造一段长文本，方便看剪枝效果
    long_text = """
    人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    人工智能的研究领域包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    机器学习是实现人工智能的一种方法，它使用算法来解析数据、从中学习，然后对真实世界中的事件做出决策和预测。
    深度学习是机器学习的一个子集，它利用多层神经网络从大量数据中学习。
    近年来，深度学习在图像识别、语音识别、自然语言处理等领域取得了突破性进展。
    人工智能技术正在深刻改变着各行各业，从医疗健康到金融服务，从教育培训到交通运输。
    """ * 10  # 重复10次，造出足够长的文本

    print("=" * 60)
    print("🔬 Token 剪枝器测试")
    print("=" * 60)

    # 测试三种策略
    for strategy in ["head_tail", "similarity", "hybrid"]:
        print(f"\n{'=' * 40}")
        print(f"📋 策略: {strategy}")
        print(f"{'=' * 40}")

        result = prune_text(long_text, strategy=strategy, target_ratio=0.85)

        print(f"  原始 Token 数: {result['original_tokens']}")
        print(f"  剪枝后 Token 数: {result['pruned_tokens']}")
        print(f"  压缩比: {result['compression_ratio']}")
        print(f"  剪枝后文本预览: {result['pruned_text'][:200]}...")

    print(f"\n{'=' * 60}")
    print("✅ 剪枝器测试完成")
    print(f"{'=' * 60}")
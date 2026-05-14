# llm_engine.py (第二周优化版)
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from loguru import logger
from typing import Dict, List, Optional

# 全局变量：避免重复加载模型
_model = None
_tokenizer = None
_model_path = "E:\code\Models\Qwen2.5-3B-Instruct\qwen\Qwen2___5-3B-Instruct"  # ← 改成你自己的路径


def _load_model():
    """懒加载模型和分词器"""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        logger.info("🔄 正在加载模型和分词器...")
        _tokenizer = AutoTokenizer.from_pretrained(_model_path)
        _model = AutoModelForCausalLM.from_pretrained(
            _model_path,
            torch_dtype=torch.float16,  # 用 float16 节省显存，RTX 5060 8GB 够用
            device_map="auto"  # 自动分配到 GPU
        )
        logger.info("✅ 模型加载完成")
    return _model, _tokenizer


def generate(prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> Dict:
    """
    标准推理生成接口 (第二周优化版)

    参数:
        prompt: 用户输入的文本
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度 (0.0 为贪婪解码, >0 增加随机性)
        top_p: 核采样阈值

    返回:
        包含 answer 和 latency 详细指标的字典
    """
    try:
        model, tokenizer = _load_model()

        # ═══════════════════════════════════════
        # 1. 输入处理
        # ═══════════════════════════════════════
        total_start = time.time()

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        input_tokens = model_inputs.input_ids.shape[1]  # 输入 token 数量

        # ═══════════════════════════════════════
        # 2. 流式生成（用 TextIteratorStreamer 拿到每个输出块的时间）
        # ═══════════════════════════════════════
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        gen_kwargs = dict(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id
        )

        # 在子线程里运行 model.generate，主线程负责收 streamer 出来的文本
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        # ═══════════════════════════════════════
        # 3. 逐块接收，记录 TTFT 和每个 token 的间隔
        # ═══════════════════════════════════════
        gen_start = time.time()
        tftt = None
        output_text = ""
        inter_token_times = []  # 用来算 TPOT

        for idx, new_text in enumerate(streamer):
            now = time.time()
            if idx == 0:
                # 第一个有意义的 token 到达 → 记录 TTFT
                tftt = now - gen_start
            else:
                # 后续每个 token → 记录与前一个 token 的时间间隔
                inter_token_times.append(now - last_token_time)

            output_text += new_text
            last_token_time = now

        gen_end = time.time()
        generation_time = gen_end - gen_start  # 纯生成耗时

        # ═══════════════════════════════════════
        # 4. 计算性能指标
        # ═══════════════════════════════════════
        # 输出 token 数（用 tokenizer 编码输出文本得到）
        output_ids = tokenizer.encode(output_text, add_special_tokens=False)
        output_tokens = len(output_ids)

        tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0

        # TPOT: 平均每个输出 token 的间隔时间（毫秒）
        tpot_mean = np.mean(inter_token_times) * 1000 if inter_token_times else 0
        tpot_median = np.median(inter_token_times) * 1000 if inter_token_times else 0
        tpot_p95 = np.percentile(inter_token_times, 95) * 1000 if inter_token_times else 0

        total_time = time.time() - total_start

        # ═══════════════════════════════════════
        # 5. 构造返回结果
        # ═══════════════════════════════════════
        result = {
            "answer": output_text,
            "latency": {
                # 第一周的 6 项核心指标
                "ttft": round(tftt, 4) if tftt else 0.0,
                "generation_time": round(generation_time, 4),
                "total_time": round(total_time, 4),
                "tokens_per_second": round(tokens_per_second, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,

                # 第二周新增：TPOT 详细指标
                "tpot_mean_ms": round(tpot_mean, 2),  # 平均 TPOT
                "tpot_median_ms": round(tpot_median, 2),  # 中位数 TPOT
                "tpot_p95_ms": round(tpot_p95, 2)  # P95 TPOT
            }
        }

        logger.info(f"📊 推理完成 | 总耗时: {total_time:.2f}s | TTFT: {tftt:.4f}s | 速度: {tokens_per_second:.2f} tok/s")
        return result

    except Exception as e:
        logger.error(f"❌ 推理出错: {str(e)}")
        return {
            "answer": "抱歉，推理过程中出现错误，请稍后重试。",
            "latency": {
                "ttft": 0.0, "generation_time": 0.0, "total_time": 0.0,
                "tokens_per_second": 0.0, "input_tokens": 0, "output_tokens": 0,
                "tpot_mean_ms": 0.0, "tpot_median_ms": 0.0, "tpot_p95_ms": 0.0
            }
        }


# ═══════════════════════════════════════
# 快速自测
# ═══════════════════════════════════════
if __name__ == "__main__":
    test_prompt = "请用三句话介绍机器学习。"
    result = generate(test_prompt, max_new_tokens=256)
    print("\n" + "=" * 50)
    print(f"📝 模型回答:\n{result['answer']}")
    print("\n📊 性能指标:")
    for k, v in result['latency'].items():
        print(f"  {k}: {v}")
    print("=" * 50)
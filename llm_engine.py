import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from loguru import logger
from typing import Dict

_model, _tokenizer, _model_path = None, None, "E:\code\Models\Qwen2.5-3B-Instruct\qwen\Qwen2___5-3B-Instruct"

def _load_model():
    global _model, _tokenizer
    if _model is None:
        logger.info("加载模型中...")
        _tokenizer = AutoTokenizer.from_pretrained(_model_path)
        _model = AutoModelForCausalLM.from_pretrained(_model_path, torch_dtype=torch.float16, device_map="auto")
        logger.info("模型加载完成")
    return _model, _tokenizer

def generate(prompt: str, max_new_tokens: int = 512) -> Dict:
    try:
        model, tokenizer = _load_model()
        total_start = time.time()
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        input_tokens = model_inputs.input_ids.shape[1]

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(**model_inputs, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9, do_sample=True, streamer=streamer)
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        gen_start = time.time()
        tftt = None
        output_text = ""
        for idx, new_text in enumerate(streamer):
            if idx == 0:
                tftt = time.time() - gen_start
            output_text += new_text

        gen_end = time.time()
        generation_time = gen_end - gen_start
        output_ids = tokenizer.encode(output_text, add_special_tokens=False)
        output_tokens = len(output_ids)
        tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
        total_time = time.time() - total_start

        result = {
            "answer": output_text,
            "latency": {
                "ttft": round(tftt, 4) if tftt else 0.0,
                "generation_time": round(generation_time, 4),
                "total_time": round(total_time, 4),
                "tokens_per_second": round(tokens_per_second, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
        }
        logger.info(f"推理完成 | 总耗时: {total_time:.2f}s | 速度: {tokens_per_second:.2f} tokens/s")
        return result
    except Exception as e:
        logger.error(f"推理出错: {e}")
        return {"answer": "推理出错，请重试。", "latency": {"ttft": 0.0, "generation_time": 0.0, "total_time": 0.0, "tokens_per_second": 0.0, "input_tokens": 0, "output_tokens": 0}}

if __name__ == "__main__":
    result = generate("你好，请介绍一下RAG的核心流程。", 512)
    print(result)
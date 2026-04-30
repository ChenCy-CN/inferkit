import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from loguru import logger

model_path = "E:\code\Models\Qwen2.5-3B-Instruct\qwen\Qwen2___5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

def generate_with_perf(prompt: str, max_new_tokens: int = 512):
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

    logger.info(f"\n==== 性能统计 ====")
    logger.info(f"输入token数: {input_tokens}, 输出token数: {output_tokens}")
    logger.info(f"TTFT: {tftt:.4f}s" if tftt else "TTFT: N/A")
    logger.info(f"生成耗时: {generation_time:.4f}s, 总耗时: {total_time:.4f}s")
    logger.info(f"生成速度: {tokens_per_second:.2f} tokens/s")

    return {
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

if __name__ == "__main__":
    result = generate_with_perf("请详细解释什么是RAG技术。", 512)
    logger.info(f"输出: {result['answer']}")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path ="E:\code\Models\Qwen2.5-3B-Instruct\qwen\Qwen2___5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "你好介绍下自己"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=512, temperature=0.7, top_p=0.9)
output_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

print(f"\n用户: {prompt}\n模型: {response}")
from llm_engine import generate

if __name__ == "__main__":
    prompt = input("请输入你的问题：")
    result = generate(prompt, max_new_tokens=512)
    print("\n== 模型回答 ==")
    print(result["answer"])
    print("\n== 性能统计 ==")
    for k, v in result["latency"].items():
        print(f"{k}: {v}")
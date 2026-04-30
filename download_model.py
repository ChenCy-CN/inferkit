
from modelscope import snapshot_download

# 下载Qwen2.5-3B-Instruct模型
model_dir = snapshot_download(
    "qwen/Qwen2.5-3B-Instruct",
    cache_dir="E:/code/Models/Qwen2.5-3B-Instruct",
    revision="master"
)
print(f"模型下载完成，保存路径: {model_dir}")


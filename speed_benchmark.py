# speed_benchmark.py
import time
import statistics
from llm_engine import generate
from loguru import logger

# ═══════════════════════════════════════
# 测试用例：模拟真实使用场景
# ═══════════════════════════════════════
TEST_CASES = [
    {
        "name": "短文本",
        "prompt": "你好，请用一句话介绍你自己。",
        "max_new_tokens": 128
    },
    {
        "name": "中等文本",
        "prompt": "请详细解释什么是机器学习，并说明它与传统编程的区别。",
        "max_new_tokens": 256
    },
    {
        "name": "长文本",
        "prompt": "请写一篇800字左右的文章，详细介绍人工智能从1950年代到2020年代的发展历程，包括关键人物、重要事件和技术突破。",
        "max_new_tokens": 512
    }
]

# 每个用例重复跑多少次（取平均值更稳定）
REPEAT_COUNT = 3


def run_benchmark():
    """运行完整的基准测试"""
    logger.info("=" * 60)
    logger.info("🚀 开始 LLM 推理速度基准测试")
    logger.info("=" * 60)

    all_results = []

    for case in TEST_CASES:
        logger.info(f"\n📋 测试用例: {case['name']}")
        logger.info(f"   Prompt 长度: {len(case['prompt'])} 字符")
        logger.info(f"   Max New Tokens: {case['max_new_tokens']}")
        logger.info(f"   重复次数: {REPEAT_COUNT}")

        case_results = {
            "name": case["name"],
            "ttft_list": [],
            "tps_list": [],
            "total_time_list": [],
            "output_tokens_list": []
        }

        # 重复跑多次，取平均值
        for i in range(REPEAT_COUNT):
            logger.info(f"   第 {i + 1}/{REPEAT_COUNT} 次推理...")

            result = generate(
                case["prompt"],
                max_new_tokens=case["max_new_tokens"]
            )

            latency = result["latency"]

            # 收集各项指标
            case_results["ttft_list"].append(latency["ttft"])
            case_results["tps_list"].append(latency["tokens_per_second"])
            case_results["total_time_list"].append(latency["total_time"])
            case_results["output_tokens_list"].append(latency["output_tokens"])

        # ═══════════════════════════════════════
        # 汇总这个用例的统计结果
        # ═══════════════════════════════════════
        summary = {
            "name": case["name"],
            "ttft_avg": statistics.mean(case_results["ttft_list"]),
            "ttft_min": min(case_results["ttft_list"]),
            "ttft_max": max(case_results["ttft_list"]),
            "tokens_per_second_avg": statistics.mean(case_results["tps_list"]),
            "total_time_avg": statistics.mean(case_results["total_time_list"]),
            "output_tokens_avg": statistics.mean(case_results["output_tokens_list"])
        }

        all_results.append(summary)

        # 打印这个用例的汇总
        logger.info(f"   ✅ {case['name']} 汇总:")
        logger.info(f"      TTFT 平均: {summary['ttft_avg']:.4f}s")
        logger.info(f"      平均生成速度: {summary['tokens_per_second_avg']:.2f} tokens/s")
        logger.info(f"      平均总耗时: {summary['total_time_avg']:.2f}s")
        logger.info(f"      平均输出 token 数: {summary['output_tokens_avg']:.0f}")

    # ═══════════════════════════════════════
    # 打印最终汇总报告
    # ═══════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("📊 最终基准测试报告")
    logger.info("=" * 60)

    for r in all_results:
        logger.info(f"\n🔹 {r['name']}:")
        logger.info(f"   TTFT:          {r['ttft_avg']:.4f}s (范围 {r['ttft_min']:.4f}s ~ {r['ttft_max']:.4f}s)")
        logger.info(f"   生成速度:      {r['tokens_per_second_avg']:.2f} tokens/s")
        logger.info(f"   总耗时:        {r['total_time_avg']:.2f}s")
        logger.info(f"   输出 Token 数: {r['output_tokens_avg']:.0f}")

    logger.info("\n✅ 基准测试完成")

    return all_results


if __name__ == "__main__":
    run_benchmark()
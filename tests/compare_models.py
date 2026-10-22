"""3b / 7b 对比：在 7b 跑过的那批题上，对比两个模型各两次运行的结果，生成 results/model_comparison_v2.md。

输入：results/eval_v2_3b.json、eval_v2_3b_t050_run2.json（3b，阈值 0.50 两次全量）
      results/eval_v2_7b.json、eval_v2_7b_run2.json（7b，阈值 0.50，只跑了部分题）
只比较 7b 跑过的题；"复核后"口径来自 tests/review_v2_3b.json / review_v2_7b.json。

用法：python tests/compare_models.py
"""
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_runs import rate  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RUNS = [("3b 第 1 次", "eval_v2_3b.json"), ("3b 第 2 次", "eval_v2_3b_t050_run2.json"),
        ("7b 第 1 次", "eval_v2_7b.json"), ("7b 第 2 次", "eval_v2_7b_run2.json")]
GROUPS = [
    ("table_tier", lambda r: r["category"] == "table_tier"),
    ("confusable", lambda r: r["category"] == "confusable"),
    ("其他有答案题（3b 答错过的）", lambda r: r["answerable"] and r["category"] not in ("table_tier", "confusable")),
    ("negative_near（3b 编过答案的）", lambda r: not r["answerable"]),
]


def percentile_of(values: list[float], p: float) -> float:
    """最近秩法百分位数（和 eval.py 口径一致）。"""
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)] if s else 0.0


def mark(r: dict) -> str:
    """单题结论简写：✅ 对 / ❌ 错，错的标原因。"""
    if r["final_correct"]:
        return "✅"
    if not r["answerable"]:
        return "❌ 编答案"
    return "❌ 误拒" if r["fallback"] else "❌ 答错"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    runs = []
    for label, fname in RUNS:
        p = RESULTS_DIR / fname
        if not p.exists():
            print(f"skip missing {fname}")
            continue
        runs.append((label, {r["id"]: r for r in json.loads(p.read_text(encoding="utf-8"))["records"]}))
    ids_7b = [i for i in runs[0][1] if all(i in recs for _, recs in runs)]
    lines = ["# 3b / 7b 对比（v2，阈值 0.50）", "",
             f"> 由 `python tests/compare_models.py` 生成。只比较 7b 跑过的 {len(ids_7b)} 题："
             "table_tier + confusable 全部、3b 在 0.50 两次运行中答错过的题、3b 在任一次运行中编过答案的 negative_near 题。"
             "**这批题是按'难'挑的，正确率不能代表全量 82 题的水平。**各模型都跑了两次，用来区分模型差距和随机波动。", "",
             "## 1. 分组正确率（复核后）", "",
             "| 分组 | n | " + " | ".join(label for label, _ in runs) + " |",
             "|---|---|" + "---|" * len(runs)]
    for gname, pred in GROUPS + [("合计", lambda r: True)]:
        gids = [i for i in ids_7b if pred(runs[0][1][i])]
        cells = [rate(sum(recs[i]["final_correct"] for i in gids), len(gids)) for _, recs in runs]
        lines.append(f"| {gname} | {len(gids)} | " + " | ".join(cells) + " |")

    lines += ["", "## 2. 错误类型与耗时", "",
              "| 运行 | 误拒（有答案却兜底） | 答错（有答案） | 编答案（无答案题） | 模型耗时中位数 | P90 | 合计 |",
              "|---|---|---|---|---|---|---|"]
    for label, recs in runs:
        rs = [recs[i] for i in ids_7b]
        times = [r["llm_s"] for r in rs if r["called_llm"]]
        lines.append(f"| {label} | {sum(r['answerable'] and r['fallback'] for r in rs)} | "
                     f"{sum(r['answerable'] and not r['fallback'] and not r['final_correct'] for r in rs)} | "
                     f"{sum((not r['answerable']) and not r['final_correct'] for r in rs)} | "
                     f"{statistics.median(times):.2f} 秒 | {percentile_of(times, 90):.2f} 秒 | {sum(times):.0f} 秒 |")

    lines += ["", "## 3. 逐题结论", "", "| 题号 | 类别 | 问题 | " + " | ".join(label for label, _ in runs) + " |",
              "|---|---|---|" + "---|" * len(runs)]
    for i in ids_7b:
        r0 = runs[0][1][i]
        lines.append(f"| {i} | {r0['category']} | {r0['question']} | " + " | ".join(mark(recs[i]) for _, recs in runs) + " |")

    lines += ["", "## 4. 7b 的回答（逐题，便于对照）", ""]
    for i in ids_7b:
        for label, recs in runs[2:]:
            r = recs[i]
            rv = r.get("review")
            note = f"（复核：{rv['verdict']} —— {rv['comment']}）" if rv and not rv["stale"] else ""
            lines.append(f"- **{i}** {label}：{mark(r)} {r['answer']}{note}")
    out = RESULTS_DIR / "model_comparison_v2.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

"""阈值校准：对比不同阈值（以及同一阈值重跑）的评估结果，生成 results/threshold_calibration_v2.md。

输入：results/ 下 eval.py 生成的 eval_v2_3b*.json（每个文件是一次全量运行）
输出：
    1. 有答案题 / 无答案题的 top1 相似度分布（直方图 + 最低/最高的题）
    2. 各次运行的指标对比表（第 1 层拦截量、误拒率、正确率、幻觉、模型调用次数和耗时）
    3. 同一阈值重跑两次的波动（结论变化的题），用来判断阈值之间的差异是不是噪声
    4. 与基线（0.50 第 1 次）相比，结论发生变化的题

用法：python tests/compare_runs.py
"""
import json
import math
import statistics
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
# (文件名, 显示名)；基线放第一个
RUNS = [
    ("eval_v2_3b.json", "0.50（第 1 次，基线）"),
    ("eval_v2_3b_t050_run2.json", "0.50（第 2 次，重跑）"),
    ("eval_v2_3b_t045.json", "0.45"),
    ("eval_v2_3b_t055.json", "0.55"),
]


def rate(k: int, n: int) -> str:
    """百分比 + 分子分母。"""
    return f"{k / n * 100:.1f}% ({k}/{n})" if n else "—"


def outcome(r: dict) -> str:
    """一题的最终结论（复核后）：对 / 错，并带上兜底层级，便于看变化。"""
    ok = "对" if r["final_correct"] else "错"
    where = f"第{r['layer']}层兜底" if r["layer"] else "回答"
    return f"{ok}（{where}）"


def metrics(recs: list[dict]) -> dict:
    """一次运行的核心指标。"""
    ans = [r for r in recs if r["answerable"]]
    neg = [r for r in recs if not r["answerable"]]
    llm = [r["llm_s"] for r in recs if r["called_llm"]]
    return {
        "l1_ans": sum(r["layer"] == 1 for r in ans), "l1_neg": sum(r["layer"] == 1 for r in neg),
        "false_refusal": sum(r["fallback"] for r in ans), "n_ans": len(ans), "n_neg": len(neg),
        "ans_auto": sum(r["auto_correct"] for r in ans), "ans_final": sum(r["final_correct"] for r in ans),
        "neg_auto": sum(r["auto_correct"] for r in neg), "neg_final": sum(r["final_correct"] for r in neg),
        "halluc_auto": sum(not r["fallback"] for r in neg),
        "all_final": sum(r["final_correct"] for r in recs), "n": len(recs),
        "unreviewed_fail": sum((not r["auto_correct"]) and not (r.get("review") and not r["review"]["stale"]) for r in recs),
        "llm_calls": len(llm), "llm_median": statistics.median(llm) if llm else 0, "llm_total": sum(llm),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    runs = []
    for fname, label in RUNS:
        p = RESULTS_DIR / fname
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            runs.append((label, d["threshold"], {r["id"]: r for r in d["records"]}, fname))
        else:
            print(f"skip missing {fname}")
    base_label, _, base, _ = runs[0]
    lines = ["# 阈值重新校准（v2，qwen2.5:3b，82 题）", "",
             "> 由 `python tests/compare_runs.py` 生成；数字全部来自 results/ 下各次运行的 JSON。"
             "\"复核后\"按 `tests/review_v2_3b.json` 合并逐条复核（按回答哈希匹配）。", ""]

    # ---------- 1. top1 相似度分布（检索结果与阈值无关，用基线即可） ----------
    groups = {"有答案": [r for r in base.values() if r["answerable"]],
              "negative_near": [r for r in base.values() if r["category"] == "negative_near"],
              "negative_far": [r for r in base.values() if r["category"] == "negative_far"]}
    lines += ["## 1. top1 相似度分布", "", "| 区间 | 有答案 | negative_near | negative_far |", "|---|---|---|---|"]
    lo = math.floor(min(r["top1_score"] for r in base.values()) * 20)
    hi = math.floor(max(r["top1_score"] for r in base.values()) * 20)
    for b in range(lo, hi + 1):
        cnt = [sum(1 for r in g if math.floor(r["top1_score"] * 20) == b) for g in groups.values()]
        lines.append(f"| {b / 20:.2f}～{(b + 1) / 20:.2f} | {cnt[0]} | {cnt[1]} | {cnt[2]} |")
    lines.append("")
    for name, g in groups.items():
        s = sorted(g, key=lambda r: r["top1_score"])
        lines.append(f"- {name}（n={len(g)}）：最低 {s[0]['top1_score']:.3f}（{s[0]['id']} {s[0]['question']}），"
                     f"最高 {s[-1]['top1_score']:.3f}（{s[-1]['id']} {s[-1]['question']}）")
    ans_min = min(r["top1_score"] for r in groups["有答案"])
    neg_all = groups["negative_near"] + groups["negative_far"]
    lines += [f"- 重叠：无答案题中 top1 ≥ 有答案题最低分 {ans_min:.3f} 的有 "
              f"{sum(r['top1_score'] >= ans_min for r in neg_all)}/{len(neg_all)} 题", ""]

    # ---------- 2. 各次运行指标对比 ----------
    lines += ["## 2. 各阈值指标对比", "",
              "| 运行 | 第 1 层拦下 有答案 / 无答案 | 有答案误拒 | 有答案正确（自动） | 有答案正确（复核后） | "
              "无答案正确拒答（自动） | 无答案正确（复核后） | 全部正确（复核后） | 未复核的失败题 | 调模型次数 | 模型耗时中位数 / 合计 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, t, recs, _ in runs:
        m = metrics(list(recs.values()))
        lines.append(f"| {label} | {m['l1_ans']} / {m['l1_neg']} | {rate(m['false_refusal'], m['n_ans'])} | "
                     f"{rate(m['ans_auto'], m['n_ans'])} | {rate(m['ans_final'], m['n_ans'])} | "
                     f"{rate(m['neg_auto'], m['n_neg'])} | {rate(m['neg_final'], m['n_neg'])} | "
                     f"{rate(m['all_final'], m['n'])} | {m['unreviewed_fail']} | {m['llm_calls']} | "
                     f"{m['llm_median']:.2f} / {m['llm_total']:.0f} 秒 |")

    # ---------- 3 / 4. 与基线相比结论变化的题 ----------
    for label, t, recs, fname in runs[1:]:
        changed = [qid for qid in base if outcome(base[qid]) != outcome(recs[qid])]
        same_input = [qid for qid in changed if [c["passed"] for c in base[qid]["top"]] == [c["passed"] for c in recs[qid]["top"]]]
        lines += ["", f"## 与基线相比：{label}（`{fname}`）", "",
                  f"结论变化 {len(changed)} 题，其中 {len(same_input)} 题送进模型的片段和基线完全相同"
                  f"（变化只能来自模型本身的随机性）。", ""]
        if changed:
            lines += ["| 题号 | 类别 | top1 | 基线 | 本次 | 送进模型的片段是否相同 |", "|---|---|---|---|---|---|"]
            for qid in changed:
                r = recs[qid]
                lines.append(f"| {qid} | {r['category']} | {r['top1_score']:.3f} | {outcome(base[qid])} | {outcome(r)} | "
                             f"{'相同' if qid in same_input else '不同'} |")
    out = RESULTS_DIR / "threshold_calibration_v2.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

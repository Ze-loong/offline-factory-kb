"""评估脚本：对 tests/questions.json 每道题跑完整链路（检索 → 三层把关），按类别统计并生成报告。

输出：
    results/eval_v2_<模型>.json   每题明细 + 汇总指标（机器可读）
    results/eval_v2_<模型>.md     汇总表 + 全部失败题清单（附模型原始输出、逐条复核结论）

判分规则（自动）：
    有答案题：回答正确 = 没有兜底 且 包含全部 expected_keywords（归一化后比较）；兜底 = 误拒
    无答案题：兜底 = 正确拒答；没兜底 = 误答（幻觉）
    检索命中 = expected_source 的任意一组出现在 top3；cross_chunk 另算"两组都命中"
逐条复核：tests/review_v2_<模型>.json（按题号记录 scoring_error / real_error / acceptable + 说明），
          生成报告时合并进去；回答变了的题，旧复核自动标为"已过期"，需要重新复核。

用法：
    python eval.py                              # 全量评估（默认模型 = rag.LLM_MODEL）
    python eval.py --model qwen2.5:7b --ids V07,F01   # 只跑指定题（第 6 步 7b 对比用）
    python eval.py --report                     # 不重跑，只用已有 JSON + 复核文件重新生成 .md
    python eval.py --threshold 0.55 --suffix _t055   # 阈值校准：临时换阈值跑一遍，结果另存
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter

import rag
from tests.check_questions import CATEGORIES, NEGATIVE, keyword_hit, normalize

QUESTIONS_FILE = rag.TESTS_FILE
DIAG_TOP_K = 10      # 诊断用：额外检索前 10 名，看期望来源排第几（不影响问答链路）
LAYER = {"below_threshold": 1, "model_not_answerable": 2, "json_parse_error": 3, "empty_answer": 3}
REVIEW_TEXT = {"scoring_error": "判分误差（实际答对）", "real_error": "真实错误", "acceptable": "可接受（边界题，如实说明资料未规定）",
               "false_pass": "假通过（关键词都在，但回答里有错误或误导内容）", "pass_note": "通过（附说明）"}


def model_tag(model: str) -> str:
    """qwen2.5:3b → 3b，用于结果文件名。"""
    return model.split(":")[-1]


def source_key(chunk: dict) -> str:
    """片段来源写成"文件名#章节"，和测试集里的 expected_source 对齐。"""
    return f"{chunk['source']}#{chunk['section']}"


def percentile(values: list[float], p: float) -> float | None:
    """最近秩法求百分位数（P90 等），没有数据时返回 None。"""
    if not values:
        return None
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def answer_hash(text: str) -> str:
    """回答文本的短哈希，用来判断逐条复核是否针对的是同一个回答。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def evaluate_one(item: dict, model: str) -> dict:
    """跑一道题的完整链路，并按规则自动判分。"""
    r = rag.ask(item["question"], model)
    top3 = [source_key(c) for c in r["chunks"]]
    rec = {
        "id": item["id"], "category": item["category"], "split": item["split"], "origin": item["origin"],
        "question": item["question"],
        "answerable": item["category"] not in NEGATIVE,
        "top": [{"id": c["id"], "source": source_key(c), "score": c["score"], "passed": c["passed"],
                 "cited": c.get("cited", False)} for c in r["chunks"]],
        "top1_score": r["chunks"][0]["score"],
        "answer": r["answer"], "answer_hash": answer_hash(r["answer"]),
        "fallback": r["fallback"], "reason": r["reason"], "layer": LAYER.get(r["reason"]),
        "raw_llm_output": r.get("raw_llm_output"),
        "retrieve_s": r["timing"]["retrieve_s"], "llm_s": r["timing"]["llm_s"],
        "called_llm": r["reason"] != "below_threshold",
    }
    # 诊断：期望来源在前 DIAG_TOP_K 名里的排名（第 5 步分析"排第 4 被挤掉"之类的情况）
    diag = [source_key(c) for c in rag.retrieve(item["question"], top_k=DIAG_TOP_K)]
    rec["top3_same_section_dup"] = len(top3) != len(set(top3))
    if rec["answerable"]:
        groups = item["expected_source"]
        group_hits = [any(s in top3 for s in g) for g in groups]
        rec["expected_keywords"] = item["expected_keywords"]
        rec["expected_source"] = groups
        rec["retrieval_hit"] = any(group_hits)
        rec["retrieval_hit_all"] = all(group_hits)
        rec["expected_rank"] = [min((diag.index(s) + 1 for s in g if s in diag), default=None) for g in groups]
        ans_norm = normalize(r["answer"])
        rec["missing_keywords"] = [] if r["fallback"] else \
            [kw for kw in item["expected_keywords"] if not keyword_hit(kw, ans_norm)]
        rec["auto_correct"] = (not r["fallback"]) and not rec["missing_keywords"]
        rec["auto_outcome"] = "correct" if rec["auto_correct"] else ("false_refusal" if r["fallback"] else "wrong_answer")
    else:
        rec["auto_correct"] = r["fallback"]
        rec["auto_outcome"] = "correct_refusal" if r["fallback"] else "hallucination"
    return rec


def load_reviews(tag: str) -> dict:
    """读取逐条复核文件（不存在则为空）。"""
    path = rag.BASE_DIR / "tests" / f"review_v2_{tag}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["reviews"]


def apply_reviews(records: list[dict], reviews: dict) -> None:
    """把逐条复核结论合并到每题记录：final_correct = 自动判分，或按复核改判。"""
    for rec in records:
        # 同一题在不同轮次（换阈值、重跑）可能给出不同回答，所以一题可以有多条复核，按回答哈希匹配
        entries = reviews.get(rec["id"]) or []
        entries = entries if isinstance(entries, list) else [entries]
        match = next((e for e in entries if e.get("answer_hash") == rec["answer_hash"]), None)
        rec["review"] = None
        if match:
            rec["review"] = {**match, "stale": False}
        elif entries:
            rec["review"] = {**entries[-1], "stale": True}
        verdict = rec["review"]["verdict"] if rec["review"] and not rec["review"]["stale"] else None
        # 判分误差、边界题可接受 → 改判为正确；真实错误 → 维持错误；假通过（自动判对但内容有错）→ 改判为错误
        rec["final_correct"] = (rec["auto_correct"] or verdict in ("scoring_error", "acceptable")) and verdict != "false_pass"


def rate(k: int, n: int) -> str:
    """百分比和题数一起输出，例如 "83.3% (10/12)"。"""
    return f"{k / n * 100:.1f}% ({k}/{n})" if n else "—"


def summarize(records: list[dict]) -> dict:
    """按类别 + 汇总统计各项指标（数字全部来自 records）。"""
    out = {"by_category": {}, "overall": {}}
    for cat in CATEGORIES + ["ALL_ANSWERABLE", "ALL_NEGATIVE"]:
        if cat == "ALL_ANSWERABLE":
            rs = [r for r in records if r["answerable"]]
        elif cat == "ALL_NEGATIVE":
            rs = [r for r in records if not r["answerable"]]
        else:
            rs = [r for r in records if r["category"] == cat]
        if not rs:
            continue
        n = len(rs)
        m = {"n": n}
        if rs[0]["answerable"]:
            m["retrieval_hit"] = sum(r["retrieval_hit"] for r in rs)
            m["retrieval_hit_all"] = sum(r["retrieval_hit_all"] for r in rs)
            m["auto_correct"] = sum(r["auto_correct"] for r in rs)
            m["final_correct"] = sum(r["final_correct"] for r in rs)
            m["false_refusal"] = sum(r["fallback"] for r in rs)
            m["wrong_answer"] = sum(r["auto_outcome"] == "wrong_answer" for r in rs)
        else:
            m["correct_refusal"] = sum(r["fallback"] for r in rs)
            m["hallucination_auto"] = sum(not r["fallback"] for r in rs)
            m["final_correct"] = sum(r["final_correct"] for r in rs)
        m["layer"] = {str(k): sum(r["layer"] == k for r in rs) for k in (1, 2, 3)}
        out["by_category"][cat] = m
    llm_times = [r["llm_s"] for r in records if r["called_llm"]]
    ret_times = [r["retrieve_s"] for r in records]
    out["overall"] = {
        "n": len(records),
        "layer": {str(k): sum(r["layer"] == k for r in records) for k in (1, 2, 3)},
        "answered": sum(not r["fallback"] for r in records),
        "retrieve_s": {"median": statistics.median(ret_times), "p90": percentile(ret_times, 90), "n": len(ret_times)},
        "llm_s": {"median": statistics.median(llm_times) if llm_times else None,
                  "p90": percentile(llm_times, 90), "n": len(llm_times)},
        "top3_same_section_dup": sum(r["top3_same_section_dup"] for r in records),
    }
    return out


def summary_table(s: dict) -> list[str]:
    """把汇总指标渲染成 Markdown 表格。"""
    lines = ["| 类别 | n | 检索命中（top3） | 回答正确（自动） | 回答正确（复核后） | 误拒（兜底） | 答错（未兜底） |",
             "|---|---|---|---|---|---|---|"]
    for cat in CATEGORIES[:5] + ["ALL_ANSWERABLE"]:
        m = s["by_category"].get(cat)
        if not m:
            continue
        hit = rate(m["retrieval_hit"], m["n"])
        if cat == "cross_chunk":
            hit += f"；两组都命中 {rate(m['retrieval_hit_all'], m['n'])}"
        lines.append(f"| {cat} | {m['n']} | {hit} | {rate(m['auto_correct'], m['n'])} | "
                     f"{rate(m['final_correct'], m['n'])} | {rate(m['false_refusal'], m['n'])} | "
                     f"{rate(m['wrong_answer'], m['n'])} |")
    lines += ["", "| 类别 | n | 正确拒答（自动） | 误答 / 幻觉（自动） | 正确（复核后） | 第 1 层拦下 | 第 2 层拦下 | 第 3 层拦下 |",
              "|---|---|---|---|---|---|---|---|"]
    for cat in ["negative_near", "negative_far", "ALL_NEGATIVE"]:
        m = s["by_category"].get(cat)
        if not m:
            continue
        lines.append(f"| {cat} | {m['n']} | {rate(m['correct_refusal'], m['n'])} | {rate(m['hallucination_auto'], m['n'])} | "
                     f"{rate(m['final_correct'], m['n'])} | {m['layer']['1']} | {m['layer']['2']} | {m['layer']['3']} |")
    return lines


def write_report(data: dict, path) -> None:
    """生成人类可读的 Markdown 报告：汇总表 + 兜底层级 + 耗时 + 全部失败题清单。"""
    recs, s_all, s_test = data["records"], data["summary"]["all"], data["summary"]["test"]
    o = s_all["overall"]
    lines = [f"# 评估报告 v2（{data['model']}）", "",
             f"> 生成：{data['generated']}　阈值 {data['threshold']}　top_k {data['top_k']}　"
             f"测试集 SHA256 `{data['questions_sha256'][:16]}…`　题数 {o['n']}", "",
             "> 判分：有答案题＝未兜底且包含全部关键词；无答案题＝输出固定话术。检索命中＝期望来源（任意一组）在 top3。"
             "\"复核后\"＝失败题经逐条复核（结论和理由见 tests/review_v2_*.json），"
             "判分误差 / 边界题可接受改判为正确，假通过改判为错误，真实错误维持。", "",
             "## 1. 全部题目（dev + test）", ""] + summary_table(s_all) + [
             "", "## 2. 仅 test 集（不含第 5 步调参用的 dev 题）", ""] + summary_table(s_test) + [
             "", "## 3. 兜底层级分布（全部题目）", "",
             "| | 第 1 层（阈值，不调模型） | 第 2 层（模型 answerable=false） | 第 3 层（JSON 异常 / 空回答） | 正常回答 |",
             "|---|---|---|---|---|"]
    for label, cat in [("有答案题", "ALL_ANSWERABLE"), ("无答案题", "ALL_NEGATIVE")]:
        m = s_all["by_category"][cat]
        lines.append(f"| {label}（n={m['n']}） | {m['layer']['1']} | {m['layer']['2']} | {m['layer']['3']} | "
                     f"{m['n'] - sum(m['layer'].values())} |")
    lines.append(f"| 合计（n={o['n']}） | {o['layer']['1']} | {o['layer']['2']} | {o['layer']['3']} | {o['answered']} |")
    rt, lt = o["retrieve_s"], o["llm_s"]
    lines += ["", "## 4. 耗时（秒）", "",
              "| 阶段 | n | 中位数 | P90 |", "|---|---|---|---|",
              f"| 检索（向量化 + Chroma 查询） | {rt['n']} | {rt['median']:.2f} | {rt['p90']:.2f} |",
              f"| 模型（只统计调用了模型的题） | {lt['n']} | {lt['median']:.2f} | {lt['p90']:.2f} |",
              "", f"top3 里出现同一章节两个片段的题：{o['top3_same_section_dup']} / {o['n']}", ""]

    origin_rows = []
    for origin in ("v1", "blind", "new"):
        rs = [r for r in recs if r["origin"] == origin]
        if rs:
            origin_rows.append(f"| {origin} | {len(rs)} | {rate(sum(r['auto_correct'] for r in rs), len(rs))} | "
                               f"{rate(sum(r['final_correct'] for r in rs), len(rs))} |")
    lines += ["## 5. 按题目来源", "", "| 来源 | n | 正确（自动） | 正确（复核后） |", "|---|---|---|---|"] + origin_rows

    reviewed_pass = [r for r in recs if r["auto_correct"] and r.get("review")]
    if reviewed_pass:
        lines += ["", "## 6. 自动判分通过、但逐条复核有意见的题", ""]
        for r in reviewed_pass:
            rv = r["review"]
            stale = "（⚠️ 复核已过期）" if rv["stale"] else ""
            lines += [f"- **{r['id']}**［{r['category']}］{r['question']}：**{REVIEW_TEXT.get(rv['verdict'], rv['verdict'])}**{stale}"
                      f" —— {rv.get('comment', '')}", f"  - 回答：{r['answer']}"]

    fails = [r for r in recs if not r["auto_correct"]]
    lines += ["", f"## 7. 失败题清单（自动判分未通过，共 {len(fails)} 题）", ""]
    for r in fails:
        rv = r.get("review")
        if rv:
            tag = REVIEW_TEXT.get(rv["verdict"], rv["verdict"]) + ("（⚠️ 复核已过期：回答变了，需重新复核）" if rv["stale"] else "")
        else:
            tag = "未复核"
        lines += [f"### {r['id']}［{r['category']}］{r['question']}", "",
                  f"- 自动判分：**{r['auto_outcome']}**　兜底原因：{r['reason']}（第 {r['layer']} 层）" if r["layer"]
                  else f"- 自动判分：**{r['auto_outcome']}**",
                  f"- 逐条复核：**{tag}**" + (f" —— {rv['comment']}" if rv and rv.get("comment") else "")]
        if r["answerable"]:
            lines += [f"- 期望关键词：{json.dumps(r['expected_keywords'], ensure_ascii=False)}；缺失：{json.dumps(r['missing_keywords'], ensure_ascii=False)}",
                      f"- 期望来源：{json.dumps(r['expected_source'], ensure_ascii=False)}；检索命中：{r['retrieval_hit']}；"
                      f"期望来源在前 {DIAG_TOP_K} 名中的排名：{r['expected_rank']}"]
        lines.append("- top3：" + "；".join(f"{t['source']}（{t['score']:.3f}{'，引用' if t['cited'] else ''}"
                                            f"{'' if t['passed'] else '，低于阈值'}）" for t in r["top"]))
        lines += [f"- 最终输出：{r['answer']}", f"- 模型原始输出：`{r['raw_llm_output']}`" if r["raw_llm_output"]
                  else "- 模型原始输出：（第 1 层拦下，未调用模型）", f"- 耗时：检索 {r['retrieve_s']} 秒 · 模型 {r['llm_s']} 秒", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_data(records: list[dict], model: str, sha: str, generated: str) -> dict:
    """合并复核结论、计算 all / test 两套汇总。"""
    apply_reviews(records, load_reviews(model_tag(model)))
    return {
        "model": model, "threshold": rag.SCORE_THRESHOLD, "top_k": rag.TOP_K, "generated": generated,
        "questions_sha256": sha,
        "summary": {"all": summarize(records), "test": summarize([r for r in records if r["split"] == "test"])},
        "records": records,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=rag.LLM_MODEL)
    ap.add_argument("--ids", default="", help="comma-separated question ids (default: all)")
    ap.add_argument("--report", action="store_true", help="rebuild .md from existing JSON + review file")
    ap.add_argument("--suffix", default="", help="extra suffix for output file names")
    ap.add_argument("--threshold", type=float, default=None, help="override SCORE_THRESHOLD (threshold calibration)")
    args = ap.parse_args()
    if args.threshold is not None:
        # 阈值校准用：只改本次运行的阈值，不改 rag.py 配置区
        rag.SCORE_THRESHOLD = args.threshold

    tag = model_tag(args.model)
    base = rag.RESULTS_DIR / f"eval_v2_{tag}{args.suffix}"
    json_path, md_path = base.with_suffix(".json"), base.with_suffix(".md")

    if args.report:
        old = json.loads(json_path.read_text(encoding="utf-8"))
        rag.SCORE_THRESHOLD = old["threshold"]  # 重新生成报告时沿用那次运行的阈值
        data = build_data(old["records"], old["model"], old["questions_sha256"], old["generated"])
    else:
        raw = QUESTIONS_FILE.read_bytes()
        items = json.loads(raw)["questions"]
        if args.ids:
            wanted = set(args.ids.split(","))
            items = [q for q in items if q["id"] in wanted]
        rag.RESULTS_DIR.mkdir(exist_ok=True)
        rag.warm_up(args.model)
        records = []
        t0 = time.perf_counter()
        for i, item in enumerate(items, 1):
            rec = evaluate_one(item, args.model)
            records.append(rec)
            print(f"[{i}/{len(items)}] {rec['id']} {rec['category']:<13} top1={rec['top1_score']:.3f} "
                  f"{rec['auto_outcome']:<15} layer={rec['layer']} llm={rec['llm_s']}s", flush=True)
        print(f"Finished {len(items)} questions in {time.perf_counter() - t0:.1f}s")
        data = build_data(records, args.model, hashlib.sha256(raw).hexdigest(), time.strftime("%Y-%m-%d %H:%M:%S"))

    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(data, md_path)
    print(f"Saved {json_path.name} and {md_path.name}")
    for line in summary_table(data["summary"]["all"]):
        print(line)


if __name__ == "__main__":
    main()

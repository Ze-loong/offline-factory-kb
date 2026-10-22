"""校验测试集 tests/questions.json 的答案标注（只对照资料原文，不跑检索、不调模型）。

检查项：
1. 字段完整、id 不重复、类别合法；
2. expected_source 里的"文件名#章节"在 docs/ 里真实存在；
3. 每个 expected_keywords（任意一种说法）能在它的来源章节原文里找到（需要计算的题在 COMPUTED 里豁免）；
4. 按类别 / 来源 / dev-test 统计题数，检查任务书要求（每类 ≥ 8 题，negative_near ≥ 15 题）。

用法：python tests/check_questions.py
"""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = BASE_DIR / "tests" / "questions.json"
DOCS_DIR = BASE_DIR / "docs"

CATEGORIES = ["direct", "paraphrase", "table_tier", "cross_chunk", "confusable", "negative_near", "negative_far"]
NEGATIVE = {"negative_near", "negative_far"}
# 答案需要计算、原文里不会逐字出现的关键词：{题号: [关键词]}
COMPUTED = {"T03": ["6件"], "C05": ["300"]}


def normalize(text: str) -> str:
    """判分用的归一化：小写、去空白、范围符号统一成 ~、去掉温度和硬度单位，减少写法差异造成的误判。"""
    t = text.lower()
    t = re.sub(r"\s+", "", t)
    t = t.replace("％", "%").replace("＃", "#")
    t = re.sub(r"[～〜~\-—–－至到]", "~", t)
    t = t.replace("℃", "").replace("°c", "").replace("hrc", "")
    return t


def keyword_hit(keyword, text_norm: str) -> bool:
    """一个关键词项是否命中：字符串直接比，列表表示任意一种说法命中即可。"""
    options = keyword if isinstance(keyword, list) else [keyword]
    return any(normalize(k) in text_norm for k in options)


def load_sections() -> dict[str, str]:
    """读出 docs/ 下每份资料的每个章节原文，键为"文件名#章节"。"""
    sections = {}
    for f in sorted(DOCS_DIR.glob("*.md")):
        parts = re.split(r"^## (.+)$", f.read_text(encoding="utf-8"), flags=re.M)
        for i in range(1, len(parts), 2):
            sections[f"{f.name}#{parts[i].strip()}"] = parts[i + 1]
    return sections


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    raw = QUESTIONS_FILE.read_bytes()
    data = json.loads(raw)
    qs = data["questions"]
    sections = load_sections()
    errors = []

    ids = [q["id"] for q in qs]
    errors += [f"duplicate id {i}" for i, n in Counter(ids).items() if n > 1]
    for q in qs:
        qid, cat = q["id"], q["category"]
        if cat not in CATEGORIES:
            errors.append(f"{qid}: bad category {cat}")
        if q.get("split") not in ("dev", "test"):
            errors.append(f"{qid}: bad split")
        if cat in NEGATIVE:
            if q.get("expected_answer") != "FALLBACK" or q.get("expected_keywords"):
                errors.append(f"{qid}: negative question must have expected_answer=FALLBACK and no keywords")
            continue
        if not q.get("expected_keywords") or not q.get("expected_source"):
            errors.append(f"{qid}: answerable question needs expected_keywords and expected_source")
            continue
        if cat == "cross_chunk" and len(q["expected_source"]) < 2:
            errors.append(f"{qid}: cross_chunk needs at least 2 source groups")
        srcs = [s for group in q["expected_source"] for s in group]
        missing = [s for s in srcs if s not in sections]
        errors += [f"{qid}: source not found: {s}" for s in missing]
        text_norm = normalize("".join(sections.get(s, "") for s in srcs))
        for kw in q["expected_keywords"]:
            if kw in COMPUTED.get(qid, []):
                continue
            if not keyword_hit(kw, text_norm):
                errors.append(f"{qid}: keyword {kw!r} not found in its source sections")

    print(f"questions: {len(qs)}  sha256: {hashlib.sha256(raw).hexdigest()}")
    cat_count = Counter(q["category"] for q in qs)
    for cat in CATEGORIES:
        n_dev = sum(1 for q in qs if q["category"] == cat and q["split"] == "dev")
        print(f"  {cat:<14} n={cat_count[cat]:>3}  (dev {n_dev}, test {cat_count[cat] - n_dev})")
    print("  by origin:", dict(Counter(q["origin"] for q in qs)))
    print("  by split :", dict(Counter(q["split"] for q in qs)))
    print("  answerable:", sum(q["category"] not in NEGATIVE for q in qs),
          " negative:", sum(q["category"] in NEGATIVE for q in qs))
    errors += [f"category {c} has only {cat_count[c]} questions (< 8)" for c in CATEGORIES if cat_count[c] < 8]
    if cat_count["negative_near"] < 15:
        errors.append("negative_near has fewer than 15 questions")

    print("ERRORS:" if errors else "All checks passed.")
    for e in errors:
        print("  -", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()

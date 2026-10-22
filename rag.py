"""离线知识库 Demo —— 检索 + 三层把关 + 调本地 Ollama。

链路：问题 → bge-m3 向量化 → Chroma 检索 top_k
      → ① 相似度阈值粗筛（一个都没过：不调大模型，直接兜底）
      → ② Qwen 一次调用，同时输出 answerable 判断和回答（JSON）
      → ③ 代码根据 answerable 决定：输出回答 + 引用，或输出固定兜底话术

用法：
    python rag.py "数控铣床主轴温度正常范围是多少？"   # 问一个问题
    python rag.py --test                              # 跑 tests/questions.json 全部测试题，打印相似度分数，结果写入 results/
"""
import json
import os
import sys
import time
from pathlib import Path

# 关掉 Chroma 匿名统计（离线环境下它会尝试联网上报）
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb  # noqa: E402
import ollama  # noqa: E402
from chromadb.config import Settings  # noqa: E402

# ==================== 配置区（所有可调参数集中在这一块） ====================
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"             # 原始资料目录
DB_DIR = BASE_DIR / "chroma_db"          # Chroma 本地持久化目录
RESULTS_DIR = BASE_DIR / "results"       # 测试结果输出目录
COLLECTION = "factory_kb"                # 向量库集合名

OLLAMA_HOST = "http://127.0.0.1:11434"   # 本机 Ollama 服务地址
LLM_MODEL = "qwen2.5:3b"                 # 演示默认 3b；换成 "qwen2.5:7b" 回答更准但 CPU 上首字约慢 2.5 倍
AVAILABLE_MODELS = ["qwen2.5:3b", "qwen2.5:7b"]   # 界面下拉框可选的模型
EMBED_MODEL = "bge-m3"                # 向量模型（1024 维），和 Qwen 共用同一个 Ollama 服务

CHUNK_SIZE = 350                         # 每个片段正文最多多少字符（按章节/段落/表格切，不硬切句子；不含前面拼的"文件标题 > 章节"，入库后片段实际最长约 390 字）
CHUNK_OVERLAP = 100                      # 超长段落按句子拆分时，相邻片段最多重叠多少字符（按整句取，不从半句开始；原为 50，整句重叠下几乎总是 0，见 开发与评估记录.md）
TOP_K = 3                                # 检索返回几个片段
SCORE_THRESHOLD = 0.50                   # 余弦相似度阈值：82 题实测有答案题 top1 最低 0.527、完全无关题最高 0.456，0.45/0.50/0.55 对比后取 0.50（见 开发与评估记录.md）

TEMPERATURE = 0.1                        # 低温度，回答更稳定、少发挥
NUM_CTX = 4096                           # 模型上下文长度（token）
KEEP_ALIVE = "30m"                       # 模型在内存里常驻多久，演示期间避免反复冷启动
FALLBACK_TEXT = "资料中没有找到相关内容，建议咨询相关部门负责人"   # 兜底话术，由代码固定输出
# =======================================================================

# 直连本机 Ollama，不读系统代理环境变量（电脑配了系统代理时，断网后代理也不可用，会导致连不上本机服务）
_client = ollama.Client(host=OLLAMA_HOST, trust_env=False)

# 让模型一次输出"能否回答 + 回答 + 引用了哪几个片段"，用 JSON Schema 约束格式
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
        "cited": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answerable", "answer", "cited"],
}

SYSTEM_PROMPT = """你是恒锐模具钢有限公司的内部知识库助手，只能根据给出的【资料片段】回答员工问题。
规则：
1. 先判断资料片段里是否有能直接回答问题的内容，有则 answerable 为 true，没有则为 false。
2. 资料里没有的内容一律不要编造，也不要用常识补充；只是话题相近但没有答案的，answerable 为 false。
3. answerable 为 true 时，answer 用中文完整回答：
   - 资料里有分档条件的（如请假天数、温度范围、动火级别），先确认问题中的情况落在哪一档，再给出该档的结论；
   - 资料里是流程的，按资料的步骤顺序说明每一步由谁、在多长时间内、做什么；
   - 保留资料中的具体数字、单位和审批人；cited 填写用到的片段编号。
4. answerable 为 false 时，answer 填空字符串，cited 填空列表。
只输出 JSON。"""


def get_collection():
    """打开（不存在则创建）本地 Chroma 集合，距离度量用余弦。"""
    client = chromadb.PersistentClient(path=str(DB_DIR), settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def embed(texts: list[str]) -> list[list[float]]:
    """调用本机 Ollama 的 bge-m3 把文本转成向量。"""
    return _client.embed(model=EMBED_MODEL, input=texts, keep_alive=KEEP_ALIVE)["embeddings"]


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """检索最相似的 top_k 个片段，返回文本、来源和相似度分数（1 - 余弦距离）。"""
    col = get_collection()
    res = col.query(query_embeddings=embed([question]), n_results=top_k)
    chunks = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        chunks.append({
            "id": cid,
            "text": doc,
            "source": meta["source"],
            "section": meta["section"],
            "score": round(1 - dist, 4),
        })
    return chunks


def build_prompt(question: str, chunks: list[dict]) -> str:
    """把通过阈值的片段编号后拼进提示词。"""
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[片段{i}]（来源：{c['source']} · {c['section']}）\n{c['text']}")
    return "【资料片段】\n" + "\n\n".join(parts) + f"\n\n【问题】{question}"


def search(question: str) -> dict:
    """检索 + 第 1 层阈值粗筛（不调大模型，界面可以先把这一步的结果显示出来）。"""
    t0 = time.perf_counter()
    all_chunks = retrieve(question)
    for c in all_chunks:
        c["passed"] = c["score"] >= SCORE_THRESHOLD
        c["cited"] = False
    return {
        "question": question,
        "chunks": all_chunks,
        "timing": {"retrieve_s": round(time.perf_counter() - t0, 2), "llm_s": 0.0},
    }


def answer(result: dict, model: str | None = None) -> dict:
    """第 2、3 层：模型判断 answerable 并作答，代码决定输出回答还是固定话术。"""
    model = model or LLM_MODEL
    result = {**result, "model": model}
    question = result["question"]
    passed = [c for c in result["chunks"] if c["passed"]]
    # 第 1 层没有片段通过：不调大模型，直接兜底
    if not passed:
        return {**result, "answer": FALLBACK_TEXT, "fallback": True, "reason": "below_threshold"}

    # 第 2 层：模型一次调用，同时判断 answerable 并作答
    t1 = time.perf_counter()
    resp = _client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, passed)},
        ],
        format=ANSWER_SCHEMA,
        options={"temperature": TEMPERATURE, "num_ctx": NUM_CTX},
        keep_alive=KEEP_ALIVE,
    )
    result["timing"] = {**result["timing"], "llm_s": round(time.perf_counter() - t1, 2)}
    raw = resp["message"]["content"]
    result["raw_llm_output"] = raw

    # 第 3 层：代码兜底——解析失败、模型说答不了、回答为空，一律输出固定话术
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {**result, "answer": FALLBACK_TEXT, "fallback": True, "reason": "json_parse_error"}
    text = str(data.get("answer", "")).strip()
    if data.get("answerable") is not True:
        return {**result, "answer": FALLBACK_TEXT, "fallback": True, "reason": "model_not_answerable"}
    if not text:
        return {**result, "answer": FALLBACK_TEXT, "fallback": True, "reason": "empty_answer"}

    # 只认合法的片段编号；模型没给引用时，把通过阈值的片段都标成引用
    cited_idx = [i for i in data.get("cited", []) if isinstance(i, int) and 1 <= i <= len(passed)]
    for i in cited_idx or range(1, len(passed) + 1):
        passed[i - 1]["cited"] = True
    return {**result, "answer": text, "fallback": False, "reason": "ok"}


def ask(question: str, model: str | None = None) -> dict:
    """完整问答：检索 → 三层把关 → 返回回答、引用片段、是否兜底及各阶段耗时。"""
    return answer(search(question), model)


def warm_up(model: str | None = None) -> None:
    """预热：提前把向量模型和大模型加载进内存，避免演示时第一问冷启动。"""
    embed(["warm up"])
    _client.generate(model=model or LLM_MODEL, prompt="你好", options={"num_predict": 1}, keep_alive=KEEP_ALIVE)


def unload(model: str) -> None:
    """把模型从内存卸载（切换 3b/7b 时用，16G 内存不适合两个同时常驻）。"""
    _client.generate(model=model, prompt="", keep_alive=0)


TESTS_FILE = BASE_DIR / "tests" / "questions.json"   # 测试集（第二轮从代码里迁出，含类别、关键词、期望来源）


def load_questions() -> list[dict]:
    """读取测试集。第一轮的 14 题已原样迁入（origin=v1）。"""
    return json.loads(TESTS_FILE.read_text(encoding="utf-8"))["questions"]


def run_tests() -> None:
    """跑测试集：打印每题 top_k 相似度和最终结果，并写入 results/threshold_test.json（分类统计见 eval.py）。"""
    RESULTS_DIR.mkdir(exist_ok=True)
    warm_up()
    records = []
    for item in load_questions():
        q, cat = item["question"], item["category"]
        # expect 字段沿用第一轮的写法，方便和 threshold_test_v1.json 对比
        expect = "not_in_docs" if cat.startswith("negative") else "answerable"
        r = ask(q)
        scores = [c["score"] for c in r["chunks"]]
        top = r["chunks"][0]
        print(f"[{item['id']} {cat:<13}] top1={scores[0]:.4f} scores={scores} fallback={r['fallback']} "
              f"reason={r['reason']} llm={r['timing']['llm_s']}s")
        print(f"    Q: {q}")
        print(f"    top1 source: {top['source']} / {top['section']}")
        print(f"    A: {r['answer']}")
        records.append({"id": item["id"], "category": cat, "expect": expect, **r})
    out = RESULTS_DIR / "threshold_test.json"
    out.write_text(json.dumps({"threshold": SCORE_THRESHOLD, "model": LLM_MODEL, "records": records},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved results to {out}")


if __name__ == "__main__":
    # Windows 终端默认 GBK，强制 UTF-8 输出避免中文乱码
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_tests()
    elif len(sys.argv) > 1:
        r = ask(" ".join(sys.argv[1:]))
        print(json.dumps({k: v for k, v in r.items() if k != "raw_llm_output"}, ensure_ascii=False, indent=2))
    else:
        print(__doc__)

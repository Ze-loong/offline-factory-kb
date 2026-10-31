"""离线知识库 MCP Server（stdio 传输）。

把 rag.py 的检索和问答能力按 MCP 协议暴露出来，任何 MCP 客户端都能当工具调用：
    - 工具 search_kb：只检索，返回带来源的片段（不调大模型，约 0.5 秒）
    - 工具 ask_kb：完整问答，走 rag.py 的三层把关（相似度阈值 → 模型判断能否回答 → 代码兜底）
    - 资源 kb://documents：知识库里有哪些资料和图片（只读文件目录，不碰 Ollama）
每次工具调用都在 logs/audit.jsonl 追加一行审计记录。

离线保证：本文件不发起任何网络请求；rag.py 只连本机 127.0.0.1:11434 的 Ollama 和本地 Chroma。
stdio 传输下 stdout 是协议通道，只能放 JSON-RPC 消息，所以本文件一律不 print，
调用 rag 时也把 stdout 临时重定向到 stderr，防止第三方库的输出混进协议。

启动方式（一般不用手动启动，由 MCP 客户端作为子进程拉起）：
    python mcp_server/server.py
"""
import contextlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import anyio
from pydantic import Field

# 让本文件能导入上一级目录的 rag.py（无论从哪个工作目录启动）
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 导入 rag（会连带导入 chromadb、ollama）时也把 stdout 重定向到 stderr：
# 任何库在导入阶段 print 一行，客户端都会把它当成坏掉的 JSON-RPC 消息
with contextlib.redirect_stdout(sys.stderr):
    import rag  # noqa: E402  导入时会关掉 Chroma 匿名统计
from mcp.server.fastmcp import Context, FastMCP  # noqa: E402  mcp 1.x 的写法；2.x 已改名 MCPServer，requirements 锁 <2
from mcp.server.fastmcp.exceptions import ToolError  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

# ==================== 配置区 ====================
AUDIT_FILE = ROOT / "logs" / "audit.jsonl"   # 审计日志：一行一条 JSON，已加入 .gitignore
MAX_QUESTION_LEN = 500                       # 问题最长字符数（超过直接拒绝，写进 Tool Schema）
ALLOWED_MODELS = ("qwen2.5:3b", "qwen2.5:7b")  # 只允许本机已经测过的两个模型
# =================================================

# 兜底原因 → 被哪一层拦下（返回给客户端，方便它向用户解释）
REASON_TEXT = {
    "ok": "answered from documents",
    "below_threshold": "layer 1: no chunk reached the similarity threshold, LLM not called",
    "model_not_answerable": "layer 2: LLM judged the chunks cannot answer the question",
    "json_parse_error": "layer 3: LLM output was not valid JSON, fixed fallback returned",
    "empty_answer": "layer 3: LLM returned an empty answer, fixed fallback returned",
}

mcp = FastMCP(
    "offline-factory-kb",
    instructions=(
        "Offline knowledge base of a fictional mold-steel factory (恒锐模具钢有限公司): "
        "rules, process cards, equipment manuals, inspection photos. All data is in Chinese. "
        "Use search_kb to get source chunks; use ask_kb for a guarded answer. "
        "If ask_kb returns fallback=true, tell the user the documents do not contain the answer; do not make one up."
    ),
    log_level="WARNING",   # SDK 自己的日志写 stderr，调成 WARNING 减少刷屏
)


def _client_name(ctx: Context | None) -> str:
    """从握手信息里取客户端名字（如 claude-code），审计用；取不到就返回 unknown。"""
    try:
        return ctx.session.client_params.clientInfo.name
    except Exception:
        return "unknown"


def _audit(tool: str, args: dict, t0: float, status: str, ctx: Context | None = None, **extra) -> None:
    """追加一条审计记录。审计失败不能影响工具本身，所以吞掉所有异常。"""
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "client": _client_name(ctx),
        "tool": tool,
        "args": args,
        "status": status,            # ok / error
        "duration_s": round(time.perf_counter() - t0, 2),
        **extra,
    }
    try:
        AUDIT_FILE.parent.mkdir(exist_ok=True)
        with AUDIT_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _chunk_view(c: dict, rank: int) -> dict:
    """把 rag.py 的片段整理成返回给客户端的格式（去掉内部字段）。"""
    view = {
        "rank": rank,
        "source": c["source"],
        "section": c["section"],
        "type": c["type"],                 # text：文字资料；image：视觉模型从图片识别出的文字
        "score": c["score"],               # 余弦相似度
        "passed_threshold": c["passed"],
        "text": c["text"],
    }
    if c.get("linked"):
        view["linked_from_image"] = c.get("linked_from")   # 按图片上的文件编号补进来的片段
    return view


def _run_quiet(fn, *args, **kwargs):
    """调用 rag 函数时把 stdout 重定向到 stderr：stdio 模式下 stdout 只能放协议消息。"""
    with contextlib.redirect_stdout(sys.stderr):
        return fn(*args, **kwargs)


def _ollama_error(e: Exception) -> ToolError:
    """把连不上 Ollama 之类的底层异常转成客户端看得懂的错误（会以 isError=true 返回，不会让 Server 崩掉）。"""
    return ToolError(f"Local retrieval failed: {type(e).__name__}: {e}. "
                     f"Check that Ollama is running at {rag.OLLAMA_HOST} and the index was built (python ingest.py).")


Question = Annotated[str, Field(
    min_length=1, max_length=MAX_QUESTION_LEN,
    description="员工的问题，用中文原文传入，例如：数控铣床主轴温度正常范围是多少？",
)]


@mcp.tool(annotations=ToolAnnotations(title="Search factory knowledge base", readOnlyHint=True, openWorldHint=False))
async def search_kb(
    ctx: Context,   # FastMCP 自动注入的会话上下文，不会出现在 Tool Schema 里
    question: Question,
    include_images: Annotated[bool, Field(description="是否同时检索图片片段（设备示意图、点检表照片的识别结果）")] = True,
) -> dict:
    """在离线工厂知识库里检索与问题最相关的片段，返回来源文件、章节、相似度和正文。

    只做检索、不调大模型，速度快（约 0.5 秒）。passed_threshold=false 的片段相似度低于阈值，仅供参考。
    需要一个经过把关的最终回答时，用 ask_kb。
    """
    t0 = time.perf_counter()
    args = {"question": question, "include_images": include_images}
    try:
        # rag 是同步阻塞代码，放到线程里跑，避免卡住 Server 的事件循环（期间还要响应客户端的其他消息）
        r = await anyio.to_thread.run_sync(lambda: _run_quiet(rag.search, question, use_images=include_images))
    except Exception as e:
        _audit("search_kb", args, t0, "error", ctx, error=str(e)[:200])
        raise _ollama_error(e)
    chunks = [_chunk_view(c, i) for i, c in enumerate(r["chunks"], start=1)]
    passed = sum(c["passed_threshold"] for c in chunks)
    _audit("search_kb", args, t0, "ok", ctx, passed=passed,
           sources=[f"{c['source']}#{c['section']}" for c in chunks if c["passed_threshold"]])
    return {
        "question": question,
        "threshold": rag.SCORE_THRESHOLD,
        "passed_count": passed,
        "chunks": chunks,
        "retrieve_s": r["timing"]["retrieve_s"],
    }


@mcp.tool(annotations=ToolAnnotations(title="Ask factory knowledge base", readOnlyHint=True, openWorldHint=False))
async def ask_kb(
    ctx: Context,
    question: Question,
    model: Annotated[Literal["qwen2.5:3b", "qwen2.5:7b"], Field(
        description="本机 Ollama 模型。3b 约 7～12 秒；7b 更准但 CPU 上约 20～30 秒")] = "qwen2.5:3b",
) -> dict:
    """用本机大模型回答问题，走三层把关：相似度阈值粗筛 → 模型判断资料能否回答 → 代码兜底。

    资料里没有答案时返回 fallback=true 和固定话术，reason 说明被哪一层拦下；此时不要自行编造答案。
    citations 是回答实际引用的片段（含来源文件和章节）。
    """
    t0 = time.perf_counter()
    args = {"question": question, "model": model}
    if model not in ALLOWED_MODELS:   # Schema 已经限制，这里再做一次服务端校验（客户端不一定遵守 Schema）
        _audit("ask_kb", args, t0, "error", ctx, error="model not allowed")
        raise ToolError(f"Model not allowed: {model}. Allowed: {', '.join(ALLOWED_MODELS)}")
    try:
        r = await anyio.to_thread.run_sync(lambda: _run_quiet(rag.ask, question, model=model))
    except Exception as e:
        _audit("ask_kb", args, t0, "error", ctx, error=str(e)[:200])
        raise _ollama_error(e)
    cited = [_chunk_view(c, i) for i, c in enumerate(r["chunks"], start=1) if c.get("cited")]
    _audit("ask_kb", args, t0, "ok", ctx, fallback=r["fallback"], reason=r["reason"],
           cited=[f"{c['source']}#{c['section']}" for c in cited])
    return {
        "question": question,
        "model": model,
        "answer": r["answer"],
        "fallback": r["fallback"],
        "reason": r["reason"],
        "reason_detail": REASON_TEXT.get(r["reason"], r["reason"]),
        "citations": cited,
        "timing": r["timing"],
    }


@mcp.resource("kb://documents", name="documents", mime_type="application/json",
              description="知识库收录的资料清单：文字资料的标题和文件编号，图片资料的文件名")
def list_documents() -> str:
    """列出 docs/ 下的资料和 docs/images/ 下的图片。只读目录，不调用 Ollama，也不读向量库。"""
    docs = []
    for p in sorted(rag.DOCS_DIR.glob("*.md")):
        head = p.read_text(encoding="utf-8").splitlines()[:5]
        title = next((l.lstrip("# ").strip() for l in head if l.startswith("# ")), p.stem)
        code = next((m.group(1) for l in head if (m := re.search(r"文件编号：(\S+)", l))), "")
        docs.append({"file": p.name, "title": title, "doc_code": code})
    images = [p.name for p in sorted(rag.IMAGES_DIR.glob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return json.dumps({"documents": docs, "images": images}, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")

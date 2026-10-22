"""把 docs/ 下的资料解析、切分、向量化后写入本地 Chroma（每次运行全量重建）。

切分策略：
1. 先按 Markdown 二级标题（##）切成章节，章节标题作为元数据，回答时用来显示"引用了哪一段"；
2. 章节内按空行分成段落块，文字行和表格行紧挨着的再拆开；表格整块保留（表格拆开会丢表头，模型就看不懂数字对应哪一列）；
3. 把块依次拼到不超过 CHUNK_SIZE；单个块超长时：表格按行拆并重复表头，普通段落按句子拆，相邻片段以整句重叠（不从半句开始）；
4. 每个片段前面拼上"文件标题 > 章节"，让向量里带上上下文（例如"主轴温度"属于"数控铣床"）。

用法：python ingest.py
"""
import re
import sys
import time

import chromadb
from chromadb.config import Settings

from rag import CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION, DB_DIR, DOCS_DIR, embed


def split_sentences(text: str, size: int) -> list[str]:
    """把段落拆成句子；单句本身超过 size 的，再按逗号/顿号拆，仍超长才硬切。"""
    sentences = []
    for s in re.split(r"(?<=[。；！？\n])", text):
        if not s.strip():
            continue
        if len(s) <= size:
            sentences.append(s)
            continue
        cur = ""
        for part in re.split(r"(?<=[，、,])", s):
            while len(part) > size:  # 连逗号都没有的超长内容，只能按长度硬切
                sentences.append(part[:size])
                part = part[size:]
            if cur and len(cur) + len(part) > size:
                sentences.append(cur)
                cur = ""
            cur += part
        if cur:
            sentences.append(cur)
    return sentences


def split_long_text(text: str, size: int, overlap: int) -> list[str]:
    """按句子拆分超长段落；相邻片段的重叠以整句为单位，总长不超过 overlap，不会从半句开始。"""
    sentences = split_sentences(text, size)
    pieces, cur, carried = [], [], 0  # cur：当前片段的句子列表；carried：开头几句是从上一片段带过来的重叠
    for s in sentences:
        if len(cur) > carried and len("".join(cur)) + len(s) > size:
            pieces.append("".join(cur))
            # 从末尾往前取完整的句子作为重叠，总长不超过 overlap
            tail = []
            for prev in reversed(cur):
                if len("".join(tail)) + len(prev) > overlap:
                    break
                tail.insert(0, prev)
            cur, carried = tail, len(tail)
        if cur and len(cur) == carried and len("".join(cur)) + len(s) > size:
            # 重叠句加上新句子会超长：放弃重叠，避免产生只有重复内容的片段
            cur, carried = [], 0
        cur.append(s)
    if len(cur) > carried:
        pieces.append("".join(cur))
    return pieces


def split_table(table: str, size: int) -> list[str]:
    """超长表格按行拆分，每一份都重复表头（前两行）。"""
    lines = table.splitlines()
    header, rows = lines[:2], lines[2:]
    pieces, cur = [], []
    for row in rows:
        if cur and len("\n".join(header + cur + [row])) > size:
            pieces.append("\n".join(header + cur))
            cur = []
        cur.append(row)
    if cur:
        pieces.append("\n".join(header + cur))
    return pieces


def split_blocks(body: str) -> list[str]:
    """按空行分块；块内如果文字行和表格行紧挨着（如"下表为……："下面直接接表格），再拆成文字块和表格块。"""
    blocks = []
    for b in re.split(r"\n\s*\n", body):
        run, run_is_table = [], None
        for line in b.strip().splitlines():
            is_table = line.lstrip().startswith("|")
            if run and is_table != run_is_table:
                blocks.append("\n".join(run))
                run = []
            run.append(line)
            run_is_table = is_table
        if run:
            blocks.append("\n".join(run))
    return [b.strip() for b in blocks if b.strip()]


def chunk_section(body: str) -> list[str]:
    """把一个章节的正文切成若干片段。"""
    blocks = split_blocks(body)
    chunks, cur = [], ""
    for block in blocks:
        is_table = block.startswith("|")
        if len(block) > CHUNK_SIZE:
            # 超长块：先把手上攒的提交，再单独拆这个块
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(split_table(block, CHUNK_SIZE) if is_table else
                          split_long_text(block, CHUNK_SIZE, CHUNK_OVERLAP))
            continue
        if cur and len(cur) + len(block) + 1 > CHUNK_SIZE:
            chunks.append(cur)
            cur = ""
        cur = f"{cur}\n{block}" if cur else block
    if cur:
        chunks.append(cur)
    return chunks


def parse_markdown(path) -> list[dict]:
    """解析一份 Markdown 资料，返回带来源、章节信息的片段列表。"""
    text = path.read_text(encoding="utf-8")
    title_match = re.search(r"^# (.+)$", text, flags=re.M)
    title = title_match.group(1).strip() if title_match else path.stem
    # 按二级标题切章节；标题之前的部分（文件编号、版本等）作为"文件信息"
    parts = re.split(r"^## (.+)$", text, flags=re.M)
    sections = [("文件信息", re.sub(r"^# .+$", "", parts[0], flags=re.M))]
    sections += [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts), 2)]

    records = []
    for section, body in sections:
        for piece in chunk_section(body):
            records.append({
                "text": f"{title} > {section}\n{piece}",
                "source": path.name,
                "section": section,
            })
    return records


def main() -> None:
    """全量重建向量库。"""
    sys.stdout.reconfigure(encoding="utf-8")
    files = sorted(DOCS_DIR.glob("*.md"))
    if not files:
        print(f"No .md files found in {DOCS_DIR}")
        return

    records = []
    for f in files:
        recs = parse_markdown(f)
        print(f"Parsed {f.name}: {len(recs)} chunks, max {max(len(r['text']) for r in recs)} chars")
        records.extend(recs)

    t0 = time.perf_counter()
    vectors = embed([r["text"] for r in records])
    print(f"Embedded {len(records)} chunks in {time.perf_counter() - t0:.2f}s (dim={len(vectors[0])})")

    client = chromadb.PersistentClient(path=str(DB_DIR), settings=Settings(anonymized_telemetry=False))
    # 全量重建：先删旧集合，避免资料修改后残留旧片段
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    # 片段 id 用"文件名#序号"，同一文件内按顺序编号
    counters: dict[str, int] = {}
    ids = []
    for r in records:
        counters[r["source"]] = counters.get(r["source"], 0) + 1
        ids.append(f"{r['source']}#{counters[r['source']]}")
    col.add(
        ids=ids,
        embeddings=vectors,
        documents=[r["text"] for r in records],
        metadatas=[{"source": r["source"], "section": r["section"]} for r in records],
    )
    print(f"Stored {col.count()} chunks into collection '{COLLECTION}' at {DB_DIR}")


if __name__ == "__main__":
    main()

"""把 docs/ 下的资料解析、切分、向量化后写入本地 Chroma（每次运行全量重建）。

切分策略：
1. 先按 Markdown 二级标题（##）切成章节，章节标题作为元数据，回答时用来显示"引用了哪一段"；
2. 章节内按空行分成段落块，文字行和表格行紧挨着的再拆开；表格整块保留（表格拆开会丢表头，模型就看不懂数字对应哪一列）；
3. 把块依次拼到不超过 CHUNK_SIZE；单个块超长时：表格按行拆并重复表头，普通段落按句子拆，相邻片段以整句重叠（不从半句开始）；
4. 每个片段前面拼上"文件标题 > 章节"，让向量里带上上下文（例如"主轴温度"属于"数控铣床"）。
5. 图片：不在这里调视觉模型，只读 describe_images.py 存好的识别结果（docs/images/_desc/*.json），
   用代码按固定模板拼成一段文字（不用大模型改写，避免二次引入错误），一张图一个片段（不拆，
   否则"62℃"会和"哪台设备、哪一天"分家）。图上读到的文件编号（如 HR-SB-003）映射到 docs 里的资料，
   存进元数据 related，问答时用来做关联补检索（见 rag.py）。

用法：python ingest.py
"""
import json
import re
import sys
import time

import chromadb
from chromadb.config import Settings

from rag import CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION, DB_DIR, DOCS_DIR, IMAGE_DESC_DIR, embed


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


# ==================== 图片识别结果 → 入库文字 ====================

HEADER_KEYS = ["设备编号", "设备型号", "点检日期", "班次", "点检人", "环境温度", "炉次号"]
FOOTER_KEYS = ["异常情况说明及处理", "班组长确认", "设备员抽查"]
ITEM_PARTS = ("实测值", "判定", "备注")


def build_doc_index() -> dict[str, tuple[str, str]]:
    """从每份资料开头的"文件编号：HR-XX-000"建立 编号 → (文件名, 标题) 的映射，不手写对照表。"""
    index = {}
    for f in sorted(DOCS_DIR.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        m = re.search(r"文件编号[:：]\s*(HR-[A-Z]{2}-\d{3})", text)
        title = re.search(r"^# (.+)$", text, flags=re.M)
        if m:
            index.setdefault(m.group(1), (f.name, title.group(1).strip() if title else f.stem))
    return index


def clean_name(name: str) -> str:
    """字段名归一化：去空格、去掉模型照抄的"项目名称"前缀。"""
    name = re.sub(r"\s+", "", str(name))
    return name.removeprefix("项目名称")


def parse_form(fields: list[dict]) -> tuple[dict, list[dict], dict]:
    """把第一遍整表转写的 fields 整理成 表头 / 逐项 / 表尾。
    兼容实测见到的两种写法：
      A："主轴温度 实测值""主轴温度 判定"……（可能带"项目名称"前缀）
      B："点检项目=主轴温度""实测值 / 状态=62""备注=…"按行依次出现
    """
    header, footer, rows = {}, {}, []
    current = None
    for f in fields:
        name = clean_name(f.get("name", ""))
        value = str(f.get("value", "")).strip().rstrip("。.；;")   # 去掉结尾标点，拼文字时统一加
        if name == "点检项目":                      # 写法 B：新的一行开始
            current = {"item": value, "实测值": "", "备注": ""}
            rows.append(current)
            continue
        if name in ("实测值/状态", "实测值") and current is not None:
            current["实测值"] = value
            continue
        if name == "备注" and current is not None:
            current["备注"] = value
            continue
        part = next((p for p in ITEM_PARTS if name.endswith(p)), None)
        if part and name != part:                   # 写法 A："<项目><实测值|判定|备注>"
            item = name[: -len(part)]
            row = next((r for r in rows if r["item"] == item), None)
            if row is None:
                row = {"item": item, "实测值": "", "备注": ""}
                rows.append(row)
            if part != "判定":                       # 判定不用第一遍的（整表转写时模型基本不看勾）
                row[part] = value
            continue
        for k in HEADER_KEYS:
            if name.startswith(k):
                header[k] = value
        for k in FOOTER_KEYS:
            if name.startswith(k):
                footer[k] = value
    return header, rows, footer


def render_image(desc: dict, doc_index: dict) -> tuple[str, list[str]]:
    """把一张图的识别结果拼成入库文字，返回 (正文, 关联资料文件名列表)。"""
    p = desc.get("parsed") or {}
    title = str(p.get("title", "")).strip() or desc["image"]
    refs = [r for r in desc.get("doc_refs", []) if r in doc_index]
    related = [doc_index[r][0] for r in refs]
    ref_text = "、".join(f"{r}（《{doc_index[r][1].split(' ', 1)[-1]}》）" for r in refs)
    lines = []
    if "表单" in str(p.get("image_type", "")):
        header, rows, footer = parse_form(p.get("fields", []))
        ticks = ((desc.get("tick_check") or {}).get("rows")) or []
        lines.append("[点检记录照片，由视觉模型识别，手写内容可能有识别误差] "
                     + "；".join(f"{k} {header[k]}" for k in HEADER_KEYS if header.get(k)) + "。")
        items = []
        for i, row in enumerate(rows):
            # 判定取勾选专项那一遍：行数一致时按顺序对应，否则按项目名对应
            tick = ticks[i] if len(ticks) == len(rows) else next(
                (t for t in ticks if t.get("item") and (t["item"] in row["item"] or row["item"] in t["item"])), None)
            name = (tick or {}).get("item") or row["item"]   # 勾选专项那遍的项目名更准（实测"氮气"第一遍读成了"颜气"）
            judge = (tick or {}).get("checked") or "未识别"
            s = f"{i + 1} {name}：{row['实测值'] or '（空）'}，判定 {judge}"
            if row["备注"] and row["备注"] != "（空）":
                s += f"，备注 {row['备注']}"
            items.append(s)
        lines.append("点检结果：" + "；".join(items) + "。")
        lines.append("；".join(f"{k}：{footer[k]}" for k in FOOTER_KEYS if footer.get(k)) + "。")
        if ref_text:
            lines.append(f"表单依据：{ref_text}。")
    else:
        parts = [f for f in p.get("fields", []) if re.fullmatch(r"\d+", str(f.get("name", "")).strip())]
        fig_no = next((m.group(1) for s in p.get("visible_text", [])
                       if (m := re.search(r"图号[:：]\s*(\S+)", str(s)))), "")
        lines.append("[设备结构示意图，由视觉模型识别]" + (f" 图号 {fig_no}。" if fig_no else ""))
        lines.append("部件标号：" + "；".join(f"{f['name'].strip()} {f['value']}" for f in parts) + "。")
        if ref_text:
            lines.append(f"配套文件：{ref_text}。")
    return f"图片：{desc['image']} > {title}\n" + "\n".join(lines), related


def parse_images() -> list[dict]:
    """读取全部图片识别结果，生成入库记录；识别失败（JSON 无效）的图片跳过并提示。"""
    if not IMAGE_DESC_DIR.exists():
        return []
    doc_index = build_doc_index()
    records = []
    for f in sorted(IMAGE_DESC_DIR.glob("*.json")):
        desc = json.loads(f.read_text(encoding="utf-8"))
        if not desc.get("json_ok"):
            print(f"Skip image (recognition failed): {desc.get('image')}")
            continue
        text, related = render_image(desc, doc_index)
        records.append({"text": text, "source": desc["image"], "section": "图片", "type": "image",
                        "related": ",".join(related)})
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

    image_records = parse_images()
    for r in image_records:
        print(f"Image {r['source']}: {len(r['text'])} chars, related={r['related'] or '-'}")
    records.extend(image_records)

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
        metadatas=[{"source": r["source"], "section": r["section"], "type": r.get("type", "text"),
                    "related": r.get("related", "")} for r in records],
    )
    print(f"Stored {col.count()} chunks into collection '{COLLECTION}' at {DB_DIR}")


if __name__ == "__main__":
    main()

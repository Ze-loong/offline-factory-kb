"""离线知识库 Demo 网页界面（Gradio）。

功能：提问 → 先显示检索到的片段和相似度 → 再显示回答 / 兜底话术，并标出引用了哪份资料的哪一段。
顶部实时显示网络状态，断网演示时一眼能看到"已断网仍可用"。
第三轮：检索到图片片段（点检表照片、设备示意图）时，片段卡片里带缩略图，下方"原图"区可放大查看，
方便人对照原图核实视觉模型的识别结果；按图片上的文件编号关联补充的片段单独标色；
两个开关可现场对比"用 / 不用图片""用 / 不用关联"。

用法：python app.py  →  浏览器打开 http://127.0.0.1:7860
"""
import base64
import html
import io
import os
import socket
from functools import lru_cache

# 关掉 Gradio 使用统计和版本检查（离线环境下会尝试联网）
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

import rag  # noqa: E402

# 演示用示例问题：3 个有答案、1 个改述、1 个资料里没有、1 个明显无关
EXAMPLES = [
    "数控铣床主轴温度的正常范围是多少？",
    "H13 钢淬火加热温度是多少，回火要做几次？",
    "在车间里的设备上焊接，动火要谁审批？",
    "机床突然有怪声该怎么办，要跟谁说？",
    "数控铣床的主轴电机功率是多少？",
    "Python 怎么安装？",
    "VQ-1200-01 真空炉 10 月 21 日那一炉能进行气淬吗？",
    "VMC850-03 在 10 月 22 日点检时主轴温度正常吗？",
]

# 兜底原因 → 界面上的中文说明（对应三层把关）
REASON_TEXT = {
    "below_threshold": f"第 1 层拦截：检索到的片段相似度都低于阈值 {rag.SCORE_THRESHOLD}，没有调用大模型",
    "model_not_answerable": "第 2 层判断：模型认为资料里没有答案（answerable=false），由代码输出固定话术",
    "json_parse_error": "第 3 层兜底：模型输出格式异常，由代码输出固定话术",
    "empty_answer": "第 3 层兜底：模型回答为空，由代码输出固定话术",
}

CSS = """
.kb-card {border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 10px 12px; margin-bottom: 10px;
          background: var(--background-fill-secondary);}
.kb-card.cited {border-left: 4px solid #2e9d5b;}
.kb-card.passed {border-left: 4px solid #c9a227;}
.kb-card.dropped {border-left: 4px solid #9aa0a6; opacity: 0.75;}
.kb-head {display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-weight: 600;}
.kb-tag {font-size: 12px; padding: 1px 8px; border-radius: 10px; color: #fff;}
.kb-tag.cited {background: #2e9d5b;} .kb-tag.passed {background: #c9a227;} .kb-tag.dropped {background: #80868b;}
.kb-score {font-size: 12px; font-weight: 400; color: var(--body-text-color-subdued);}
.kb-text {font-size: 13px; margin: 6px 0 0; line-height: 1.6;}
.kb-table {border-collapse: collapse; margin: 6px 0; width: 100%; font-size: 12px;}
.kb-table th, .kb-table td {border: 1px solid var(--border-color-primary); padding: 3px 6px; text-align: left;}
.kb-table th {background: var(--background-fill-primary);}
.kb-net {font-weight: 600;}
.kb-card.linked {border-left: 4px solid #7b5ea7;}
.kb-tag.linked {background: #7b5ea7;} .kb-tag.image {background: #2f6db5;}
.kb-img {display: flex; gap: 10px; align-items: flex-start; margin-top: 6px;}
.kb-img img {max-width: 220px; max-height: 220px; border: 1px solid var(--border-color-primary); border-radius: 4px;}
.kb-warn {font-size: 12px; color: #b5651d; margin-top: 4px;}
"""


def network_status() -> str:
    """检测外网是否可达（连公共 DNS 的 53 端口，1 秒超时），用于界面顶部的网络状态灯。"""
    for host in ("223.5.5.5", "114.114.114.114"):
        try:
            with socket.create_connection((host, 53), timeout=1):
                return "🟢 外网：已连接"
        except OSError:
            continue
    return "🔴 外网：已断开（本机离线运行中）"


def status_bar(model: str) -> str:
    """顶部状态栏：网络 + 当前模型 + 关键参数。"""
    return (f"<span class='kb-net'>{network_status()}</span>　｜　大模型：<b>{model}</b>（Q4 量化，CPU 推理）"
            f"　｜　向量模型：{rag.EMBED_MODEL}　｜　图片识别：{rag.VISION_MODEL}（入库前离线跑一次）"
            f"　｜　阈值 {rag.SCORE_THRESHOLD} · top_k {rag.TOP_K}")


@lru_cache(maxsize=32)
def thumb_data_uri(name: str) -> str:
    """图片缩略图（长边 360 像素）转成 data URI 直接嵌进卡片，不依赖网络和静态文件服务。"""
    path = rag.IMAGES_DIR / name
    if not path.exists():
        return ""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((360, 360))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def gallery(items: list | None = None):
    """"原图"区：有图片才显示，没有图片时整块隐藏（否则会留一个空的大框）。"""
    return gr.Gallery(value=items or [], visible=bool(items))


def gallery_items(chunks: list[dict], answered: bool) -> list[tuple[str, str]]:
    """下方"原图"区：通过阈值的图片片段对应的原图（可点开放大），回答引用的排前面。"""
    imgs = [c for c in chunks if c.get("type") == "image" and c["passed"]]
    imgs.sort(key=lambda c: not (answered and c.get("cited")))
    out = []
    for c in imgs:
        path = rag.IMAGES_DIR / c["source"]
        if path.exists():
            mark = "回答引用 · " if answered and c.get("cited") else ""
            out.append((str(path), f"{mark}{c['source']}（视觉模型识别，请对照原图核实）"))
    return out


def text_to_html(text: str) -> str:
    """片段正文转 HTML：Markdown 表格渲染成真正的表格，其余文字保留换行。"""
    out, rows = [], []

    def flush_table():
        """把攒下的表格行输出成 <table>，第一行当表头。"""
        if not rows:
            return
        head, body = rows[0], rows[1:]
        th = "".join(f"<th>{html.escape(c)}</th>" for c in head)
        trs = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in body)
        out.append(f"<table class='kb-table'><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>")
        rows.clear()

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            # 跳过 |---|---| 分隔行
            if not all(set(c) <= set("-: ") for c in cells):
                rows.append(cells)
            continue
        flush_table()
        out.append(f"<div>{html.escape(line)}</div>")
    flush_table()
    return "".join(out)


def render_chunks(chunks: list[dict], answered: bool) -> str:
    """把检索片段渲染成卡片：已引用 / 过阈值未引用 / 未过阈值。"""
    if not chunks:
        return ""
    cards = []
    for i, c in enumerate(chunks, start=1):
        if answered and c.get("cited"):
            kind, tag = "cited", "回答引用"
        elif c.get("linked"):
            kind, tag = "linked", "关联补充"
        elif c["passed"]:
            kind, tag = "passed", "通过阈值"
        else:
            kind, tag = "dropped", "低于阈值，已丢弃"
        extra_tags = ""
        if c.get("type") == "image":
            extra_tags += "<span class='kb-tag image'>图片识别</span>"
        if c.get("linked") and kind != "linked":
            extra_tags += "<span class='kb-tag linked'>关联补充</span>"
        note = ""
        if c.get("linked"):
            note = (f"<div class='kb-warn'>按图片 {html.escape(c.get('linked_from', ''))} 上印的文件编号，"
                    f"在对应资料里补检索得到（不经过相似度阈值）</div>")
        body = f"<div class='kb-text'>{text_to_html(c['text'])}</div>"
        if c.get("type") == "image":
            uri = thumb_data_uri(c["source"])
            img = f"<img src='{uri}' alt='{html.escape(c['source'])}'>" if uri else ""
            body = (f"<div class='kb-img'>{img}<div>{body}"
                    f"<div class='kb-warn'>⚠️ 以上文字由视觉模型从图片识别，手写内容可能有误，请对照原图（下方\"原图\"区可放大）</div>"
                    f"</div></div>")
        cards.append(
            f"<div class='kb-card {kind}'>"
            f"<div class='kb-head'><span class='kb-tag {kind}'>{tag}</span>{extra_tags}"
            f"<span>#{i} {html.escape(c['source'])} · {html.escape(c['section'])}</span>"
            f"<span class='kb-score'>相似度 {c['score']:.3f}</span></div>"
            f"{note}{body}</div>"
        )
    return "".join(cards)


def on_ask(question: str, model: str, use_images: bool, link: bool):
    """提问回调（生成器）：先显示检索结果，再显示模型回答。"""
    question = (question or "").strip()
    if not question:
        yield "请输入问题。", "", "", gallery(), status_bar(model)
        return
    try:
        yield "⏳ 正在检索知识库……", "", "", gallery(), status_bar(model)
        result = rag.search(question, use_images=use_images, link=use_images and link)
        n_passed = sum(c["passed"] for c in result["chunks"])
        n_linked = sum(bool(c.get("linked")) for c in result["chunks"])
        hint = (f"⏳ 检索完成（{n_passed} 个片段进入上下文，其中关联补充 {n_linked} 个），{model} 正在本机 CPU 上生成回答……"
                if n_passed else "检索完成，没有片段通过阈值。")
        yield hint, render_chunks(result["chunks"], answered=False), "", gallery(gallery_items(result["chunks"], False)), status_bar(model)

        result = rag.answer(result, model)
    except Exception as e:  # Ollama 没启动、模型没拉取等情况
        yield f"❌ 调用本地模型失败：{html.escape(str(e))}\n\n请确认 Ollama 已启动（任务栏有 Ollama 图标）。", "", "", gallery(), status_bar(model)
        return

    t = result["timing"]
    timing = f"检索 {t['retrieve_s']} 秒 · 模型 {t['llm_s']} 秒 · 模型：{result['model']}"
    if result["fallback"]:
        md = f"### ⚠️ {result['answer']}\n\n<small>{REASON_TEXT.get(result['reason'], result['reason'])}</small>"
    else:
        cited = [c for c in result["chunks"] if c.get("cited")]
        refs = "；".join(f"{c['source']} · {c['section']}" for c in cited)
        md = f"### ✅ 依据资料回答\n\n{result['answer']}\n\n**引用**：{refs}"
        if any(c.get("type") == "image" for c in cited):
            md += "\n\n<small>⚠️ 回答用到了图片识别内容，手写字可能识别有误，请对照下方原图核实。</small>"
    yield (md, render_chunks(result["chunks"], answered=not result["fallback"]), timing,
           gallery(gallery_items(result["chunks"], not result["fallback"])), status_bar(model))


def on_model_change(model: str):
    """切换模型：卸载其他模型、预热新模型（16G 内存不适合 3b 和 7b 同时常驻）。"""
    for m in rag.AVAILABLE_MODELS:
        if m != model:
            try:
                rag.unload(m)
            except Exception:
                pass
    rag.warm_up(model)
    return status_bar(model)


with gr.Blocks(title="恒锐知识库 · 离线问答") as demo:
    gr.Markdown("## 恒锐模具钢 · 内部知识库问答（离线版 Demo）\n"
                "本机 Ollama 运行 Qwen2.5 量化模型 + bge-m3 向量 + Chroma 向量库，**全程不联网**。"
                "设备示意图和手写点检表照片由本地视觉模型 Qwen3-VL 在入库前识别成文字。资料为虚构演示数据。")
    status = gr.HTML(status_bar(rag.LLM_MODEL))
    with gr.Row():
        question = gr.Textbox(label="问题", placeholder="例如：数控铣床主轴温度的正常范围是多少？", scale=5, lines=1)
        model = gr.Dropdown(rag.AVAILABLE_MODELS, value=rag.LLM_MODEL, label="大模型", scale=1)
    with gr.Row():
        use_images = gr.Checkbox(value=rag.USE_IMAGES, label="检索图片资料（示意图 / 点检表照片）")
        link = gr.Checkbox(value=rag.LINK_EXPAND, label="命中图片时按图上文件编号关联补检索")
    with gr.Row():
        btn = gr.Button("提问", variant="primary")
        refresh = gr.Button("刷新网络状态")
    gr.Examples(EXAMPLES, inputs=question, label="示例问题")
    answer_md = gr.Markdown()
    timing_md = gr.Markdown()
    gallery_box = gr.Gallery(label="原图（点开可放大，对照核实识别结果）", columns=4, height="auto", show_label=True,
                             visible=False)
    gr.Markdown("#### 检索到的资料片段（绿色＝回答引用，黄色＝通过阈值，紫色＝按图片文件编号关联补充，灰色＝低于阈值被丢弃）")
    chunks_html = gr.HTML()

    outputs = [answer_md, chunks_html, timing_md, gallery_box, status]
    inputs = [question, model, use_images, link]
    btn.click(on_ask, inputs=inputs, outputs=outputs)
    question.submit(on_ask, inputs=inputs, outputs=outputs)
    model.change(on_model_change, inputs=model, outputs=status)
    refresh.click(status_bar, inputs=model, outputs=status)


if __name__ == "__main__":
    print("Warming up models (first load may take a few seconds)...")
    try:
        rag.warm_up()
    except Exception as e:  # Ollama 没启动、模型目录不对等情况，给出可读的排查提示
        print(f"[ERROR] Warm-up failed: {e}")
        print("Check: 1) Ollama is running (tray icon)  2) `ollama list` shows "
              f"{rag.LLM_MODEL} and {rag.EMBED_MODEL}  3) OLLAMA_MODELS points to the folder holding the models")
        raise SystemExit(1)
    print("Models ready. Open http://127.0.0.1:7860")
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        ssr_mode=False,
        inbrowser=True,  # 启动后自动打开浏览器
        allowed_paths=[str(rag.IMAGES_DIR)],  # 允许"原图"区读取 docs/images 下的图片
        theme=gr.themes.Soft(font=["Microsoft YaHei", "PingFang SC", "sans-serif"], font_mono=["Consolas", "monospace"]),
        css=CSS,
    )

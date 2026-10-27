"""用本地视觉模型识别 docs/images/ 下的图片，结果存成 JSON，供 ingest.py 入库。

表单照片跑两遍：第一遍整表转写（数值、文字），第二遍只问"判定"列勾在哪个方框（整表转写时模型基本不看勾）。

为什么单独一步：视觉模型在 CPU 上每张图要 1～2 分钟，只在图片新增或变化时跑一次；
识别结果存成文件后，入库可复现，也方便逐条核对"模型把哪里认错了"。

输出：docs/images/_desc/<图片文件名>.json
     （模型原始输出、解析后的字段、图片里读到的文件编号、模型名、参数、耗时、图片哈希）
已有结果且图片没变就跳过；--force 全部重跑。

用法：
    python describe_images.py                       # 识别 docs/images/ 下的新图片
    python describe_images.py --force               # 全部重跑
    python describe_images.py --src DIR --out DIR   # 识别别的目录（如对比实验用的图片），结果写到指定目录
"""
import argparse
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path

import ollama
from PIL import Image, ImageOps

from rag import IMAGE_DESC_DIR, IMAGES_DIR, OLLAMA_HOST, VISION_MODEL

# ==================== 配置区 ====================
MAX_SIDE = 1280        # 长边缩放到多少像素（手机原图 4000 像素以上，直接送会让图片 token 暴增、CPU 上很慢）
TEMPERATURE = 0.1      # 低温度，识别结果更稳定
NUM_CTX = 8192         # 上下文：图片约 1300～1700 token + 提示词 + 输出
NUM_PREDICT = 2000     # 输出上限，防止模型陷入重复时一直生成
PROMPT_VERSION = "v3+tick1"  # 改提示词时同步改版本号，结果文件里会记录（v3 整表转写 + tick1 勾选专项）
IMAGE_EXTS = (".png", ".jpg", ".jpeg")
# =================================================

PROMPT = """你是工厂文档数字化助手。请仔细查看这张图片，把图片里能看到的内容如实转写成 JSON。
要求：
1. image_type：图片类型，从"设备结构图""表单照片""其他"中选一个。
2. title：图片里的标题原文；没有就填空字符串。
3. visible_text：图片里所有能看清的文字，按从上到下、从左到右的顺序逐条列出，保持原文，包括手写字。
4. fields：图片里的"名称—内容"对应关系，逐条列出。
   - 设备结构图：name 填标号（如"1"），value 填该标号对应的部件名称。
   - 表单：表头栏目 name 直接用栏目名（如"设备编号""点检人"）。
     每个点检项目写三条，name 用"项目名称 + 空格 + 实测值 / 判定 / 备注"，例如"主轴温度 实测值""主轴温度 判定""主轴温度 备注"；
     "点检方法"一列是印刷说明，不用列。
   - 判定栏里印着"□正常"和"□异常"两个选项，手写对勾画在哪个方框上就填哪个（"正常"或"异常"）。
     每一行都要单独看勾的位置，不同行的结果可能不一样，不要默认填"正常"；两个都没勾填"（空）"，看不清填"[无法辨认]"。
   - 空着的栏目 value 填"（空）"。
5. 看不清的字写"[无法辨认]"，不要猜；有涂改、涂黑的地方写"[涂改，无法辨认]"；图片里没有的内容一律不要补充。
6. doc_refs：图片里出现的文件编号（如"HR-SB-003"），按原文列出；没有就填空列表。
7. summary：用两三句话说明这张图是什么、主要内容是什么。
只输出 JSON。"""

# 勾选专项提示词：整表转写时 4b 基本不看勾（3 版提示词"异常"只认出 0～1/6），单独问判定列时 4 张表 21/21。
# 所以表单照片多跑一遍这个窄问题，判定以它为准。
TICK_PROMPT = """这是一张设备点检表的照片。只看表格里"判定"这一列，其他内容都不用管。
每一行的判定格里印着两个选项"□正常"和"□异常"，点检人用笔在其中一个方框上打了对勾。
请从上到下逐行判断：对勾打在"正常"的方框上，还是"异常"的方框上。
注意：每一行要单独看，不同行的结果可能不同；对勾的起笔一般落在被选中的方框里，笔画可能会延伸到旁边的文字上。
item 填这一行的点检项目名称，checked 填"正常""异常""未勾"或"无法辨认"。只输出 JSON。"""

TICK_SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"item": {"type": "string"},
                               "checked": {"type": "string", "enum": ["正常", "异常", "未勾", "无法辨认"]}},
                "required": ["item", "checked"],
            },
        }
    },
    "required": ["rows"],
}

SCHEMA = {
    "type": "object",
    "properties": {
        "image_type": {"type": "string"},
        "title": {"type": "string"},
        "visible_text": {"type": "array", "items": {"type": "string"}},
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
                "required": ["name", "value"],
            },
        },
        "doc_refs": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["image_type", "title", "visible_text", "fields", "doc_refs", "summary"],
}

# 直连本机 Ollama，不读系统代理
client = ollama.Client(host=OLLAMA_HOST, trust_env=False, timeout=900)


def file_sha256(path: Path) -> str:
    """图片文件的哈希，用来判断图片有没有变化。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_image(path: Path) -> tuple[bytes, list, list]:
    """按 EXIF 方向摆正、长边缩到 MAX_SIDE，转成不带 EXIF 的 JPEG 字节（只在内存里，不落盘）。"""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    orig = list(im.size)
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return buf.getvalue(), orig, list(im.size)


def find_doc_refs(parsed: dict | None, raw: str) -> list[str]:
    """汇总图片里的文件编号：模型填的 doc_refs + 在全部输出文字里用正则再找一遍（防止模型漏填）。"""
    refs = []
    if parsed:
        refs += [r for r in parsed.get("doc_refs", []) if isinstance(r, str)]
    refs += re.findall(r"HR-[A-Z]{2}-\d{3}", raw or "")
    # 只保留"HR-XX-数字"形式，去掉表单编号后缀（如 HR-SB-003-R1 → HR-SB-003），去重保序
    out = []
    for r in refs:
        m = re.search(r"HR-[A-Z]{2}-\d{3}", r)
        if m and m.group(0) not in out:
            out.append(m.group(0))
    return out


def describe(path: Path) -> dict:
    """识别一张图片，返回要写进结果文件的全部内容。"""
    img_bytes, orig, sent = prepare_image(path)
    kwargs = {}
    if "thinking" in (client.show(VISION_MODEL).capabilities or []):
        kwargs["think"] = False
    t0 = time.perf_counter()
    resp = client.chat(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": PROMPT, "images": [img_bytes]}],
        format=SCHEMA,
        options={"temperature": TEMPERATURE, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
        keep_alive="10m",
        **kwargs,
    )
    wall = time.perf_counter() - t0
    raw = resp.message.content or ""
    try:
        parsed, json_ok = json.loads(raw), True
    except json.JSONDecodeError:
        parsed, json_ok = None, False

    # 表单照片：再单独问一遍判定列（勾选），入库时判定以这一遍为准
    tick = None
    if parsed and "表单" in str(parsed.get("image_type", "")):
        t1 = time.perf_counter()
        r2 = client.chat(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": TICK_PROMPT, "images": [img_bytes]}],
            format=TICK_SCHEMA,
            options={"temperature": TEMPERATURE, "num_ctx": NUM_CTX, "num_predict": 400},
            keep_alive="10m",
            **kwargs,
        )
        try:
            rows = json.loads(r2.message.content or "")["rows"]
        except (json.JSONDecodeError, KeyError, TypeError):
            rows = None
        tick = {"rows": rows, "wall_s": round(time.perf_counter() - t1, 2),
                "output_tokens": r2.eval_count, "raw": r2.message.content}
    return {
        "image": path.name,
        "image_sha256": file_sha256(path),
        "model": VISION_MODEL,
        "prompt_version": PROMPT_VERSION,
        "options": {"max_side": MAX_SIDE, "temperature": TEMPERATURE, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "orig_size": orig,
        "sent_size": sent,
        "timing": {
            "wall_s": round(wall, 2),
            "load_s": round((resp.load_duration or 0) / 1e9, 2),
            "prompt_eval_s": round((resp.prompt_eval_duration or 0) / 1e9, 2),
            "eval_s": round((resp.eval_duration or 0) / 1e9, 2),
        },
        "prompt_tokens": resp.prompt_eval_count,
        "output_tokens": resp.eval_count,
        "done_reason": resp.done_reason,
        "json_ok": json_ok,
        "doc_refs": find_doc_refs(parsed, raw),
        "parsed": parsed,
        "raw": raw,
        "tick_check": tick,   # 表单照片的勾选专项识别结果（结构图为 null）
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-run all images even if results exist")
    ap.add_argument("--src", default=str(IMAGES_DIR), help="image folder")
    ap.add_argument("--out", default=str(IMAGE_DESC_DIR), help="output folder for JSON results")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No images found in {src}")
        return
    todo = []
    for p in images:
        dst = out / f"{p.name}.json"
        if not args.force and dst.exists():
            try:
                old = json.loads(dst.read_text(encoding="utf-8"))
                if old.get("image_sha256") == file_sha256(p) and old.get("json_ok"):
                    print(f"Skip (unchanged): {p.name}")
                    continue
            except (json.JSONDecodeError, OSError):
                pass
        todo.append(p)
    if not todo:
        print("Nothing to do.")
        return

    # 先预热加载模型，后面每张图的耗时不含加载
    t0 = time.perf_counter()
    client.generate(model=VISION_MODEL, prompt="hi", options={"num_predict": 1}, keep_alive="10m")
    print(f"Model {VISION_MODEL} loaded in {time.perf_counter() - t0:.1f}s; {len(todo)} image(s) to describe")

    for i, p in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {p.name} ...", flush=True)
        try:
            rec = describe(p)
        except Exception as e:  # 单张失败不影响其他图片
            print(f"    ERROR: {e!r}")
            continue
        (out / f"{p.name}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"    wall={rec['timing']['wall_s']}s in={rec['prompt_tokens']} out={rec['output_tokens']} "
              f"json_ok={rec['json_ok']} done={rec['done_reason']} doc_refs={rec['doc_refs']}")
        if rec["tick_check"]:
            rows = rec["tick_check"]["rows"] or []
            print(f"    tick check {rec['tick_check']['wall_s']}s: " + ", ".join(r.get("checked", "?") for r in rows))

    client.generate(model=VISION_MODEL, prompt="", keep_alive=0)   # 卸载视觉模型，释放内存
    print(f"\nDone. Results in {out}")


if __name__ == "__main__":
    main()

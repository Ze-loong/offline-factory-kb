"""MCP Server 本机自测客户端：不经过任何 AI 客户端，手动按协议走一遍 stdio + JSON-RPC。

做的事和 Claude Code 连接 Server 时一样：
    1. 把 server.py 作为子进程启动（stdio 传输）
    2. initialize 握手 → 发 notifications/initialized
    3. tools/list、resources/list、resources/read
    4. tools/call search_kb，再 tools/call ask_kb（加 --no-llm 跳过 ask_kb）
每条发出 / 收到的 JSON-RPC 消息都打印出来（长内容截断），方便对照协议看。
只用标准库，全程不联网（Server 只连本机 Ollama）。

用法（在项目根目录）：
    python mcp_server/test_client.py
    python mcp_server/test_client.py --no-llm
    python mcp_server/test_client.py --question "真空炉气淬压力低于多少不能开炉？"
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SERVER = Path(__file__).resolve().parent / "server.py"
PROTOCOL_VERSION = "2025-06-18"   # 客户端声明的协议版本；Server 支持就原样返回，不支持会回它自己支持的版本
MAX_SHOW = 600                    # 打印消息时最多显示多少字符


class StdioClient:
    """最小的 MCP stdio 客户端：一行一条 JSON-RPC 消息。"""

    def __init__(self, cmd: list[str]):
        # stderr 不接管，Server 的日志直接显示在终端；stdout 只用来读协议消息
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, encoding="utf-8", bufsize=1)
        self.next_id = 1

    def _send(self, msg: dict) -> None:
        line = json.dumps(msg, ensure_ascii=False)
        print(f"\n--> {line[:MAX_SHOW]}")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict | None = None) -> dict:
        """发请求（带 id），一直读到 id 相同的响应为止；中间收到的通知照样打印。"""
        rid = self.next_id
        self.next_id += 1
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        t0 = time.perf_counter()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("Server closed stdout (crashed?). See stderr output above.")
            msg = json.loads(line)   # 这里解析失败 = Server 往 stdout 打了非协议内容
            text = json.dumps(msg, ensure_ascii=False)
            more = f" ...(+{len(text) - MAX_SHOW} chars)" if len(text) > MAX_SHOW else ""
            print(f"<-- {text[:MAX_SHOW]}{more}")
            if msg.get("id") == rid:
                print(f"    ({time.perf_counter() - t0:.2f}s)")
                if "error" in msg:
                    raise RuntimeError(f"JSON-RPC error: {msg['error']}")
                return msg["result"]

    def notify(self, method: str, params: dict | None = None) -> None:
        """发通知（没有 id，对方不回复）。"""
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def close(self) -> None:
        self.proc.stdin.close()   # 关掉 stdin，Server 读到 EOF 后自己退出
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def show_tool_result(result: dict) -> None:
    """工具结果：isError 表示工具执行失败；正常结果在 content[0].text 里（FastMCP 把返回的 dict 序列化成 JSON 文本）。"""
    text = result["content"][0]["text"]
    if result.get("isError"):
        print("    TOOL ERROR:", text)
        return
    try:
        data = result.get("structuredContent") or json.loads(text)
        print(json.dumps(data, ensure_ascii=False, indent=2)[:2500])
    except json.JSONDecodeError:
        print(text[:2500])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows 终端默认 GBK，避免中文乱码
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="数控铣床主轴温度正常范围是多少？")
    ap.add_argument("--no-llm", action="store_true", help="skip ask_kb (no LLM call)")
    a = ap.parse_args()

    client = StdioClient([sys.executable, str(SERVER)])
    try:
        print("=== 1. initialize ===")
        init = client.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "kb-test-client", "version": "0.1"},
        })
        print(f"    server={init['serverInfo']}, protocol={init['protocolVersion']}, "
              f"capabilities={list(init['capabilities'])}")
        client.notify("notifications/initialized")

        print("\n=== 2. tools/list (Tool Schema) ===")
        for t in client.request("tools/list")["tools"]:
            print(f"\n    [{t['name']}] {t.get('description', '').splitlines()[0]}")
            print("    inputSchema:", json.dumps(t["inputSchema"], ensure_ascii=False, indent=2).replace("\n", "\n    "))

        print("\n=== 3. resources/list + resources/read ===")
        res = client.request("resources/list")["resources"]
        print("    resources:", [r["uri"] for r in res])
        docs = json.loads(client.request("resources/read", {"uri": "kb://documents"})["contents"][0]["text"])
        print(f"    {len(docs['documents'])} documents, {len(docs['images'])} images")

        print(f"\n=== 4. tools/call search_kb: {a.question} ===")
        show_tool_result(client.request("tools/call", {"name": "search_kb", "arguments": {"question": a.question}}))

        print("\n=== 5. tools/call with invalid args (question too long) ===")
        show_tool_result(client.request("tools/call", {"name": "search_kb", "arguments": {"question": "x" * 600}}))

        if not a.no_llm:
            print(f"\n=== 6. tools/call ask_kb: {a.question} ===")
            show_tool_result(client.request("tools/call", {"name": "ask_kb", "arguments": {"question": a.question}}))
            print("\n=== 7. tools/call ask_kb (not in docs) ===")
            show_tool_result(client.request("tools/call", {"name": "ask_kb", "arguments": {"question": "主轴电机功率是多少千瓦？"}}))
    finally:
        client.close()
    print("\nDone. Audit log: logs/audit.jsonl")


if __name__ == "__main__":
    main()

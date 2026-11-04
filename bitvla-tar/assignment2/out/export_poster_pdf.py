#!/usr/bin/env python3
"""Export poster HTML to a single-page A0 landscape PDF via Chrome DevTools Protocol."""

from __future__ import annotations

import json
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# A0 landscape (ISO 216)
PAPER_WIDTH_IN = 1189 / 25.4
PAPER_HEIGHT_IN = 841 / 25.4


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("WebSocket closed")
        buf += chunk
    return buf


def _ws_send(sock: socket.socket, payload: str) -> None:
    data = payload.encode("utf-8")
    mask = b"\x00\x00\x00\x00"
    frame = bytearray([0x81])
    length = len(data)
    if length < 126:
        frame.append(0x80 | length)
    elif length < 65536:
        frame.append(0x80 | 126)
        frame.extend(struct.pack(">H", length))
    else:
        frame.append(0x80 | 127)
        frame.extend(struct.pack(">Q", length))
    frame.extend(mask)
    frame.extend(data)
    sock.sendall(frame)


def _ws_recv(sock: socket.socket) -> str:
    header = _recv_exact(sock, 2)
    opcode = header[0] & 0x0F
    masked = (header[1] & 0x80) != 0
    length = header[1] & 0x7F
    if length == 126:
        length = struct.unpack(">H", _recv_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack(">Q", _recv_exact(sock, 8))[0]
    if masked:
        _recv_exact(sock, 4)
    payload = _recv_exact(sock, length) if length else b""
    if opcode == 0x8:
        raise ConnectionError("WebSocket closed by peer")
    if opcode == 0x9:
        _ws_send(sock, payload.decode("utf-8", errors="replace"))
        return _ws_recv(sock)
    return payload.decode("utf-8", errors="replace")


def _cdp_call(ws_url: str, method: str, params: dict | None = None, msg_id: int = 1) -> dict:
    host_port, path = ws_url.replace("ws://", "").split("/", 1)
    host, port = host_port.split(":")
    sock = socket.create_connection((host, int(port)), timeout=120)
    try:
        key = "dGhlIHNhbXBsZSBub25jZQ=="
        req = (
            f"GET /{path} HTTP/1.1\r\n"
            f"Host: {host_port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        sock.sendall(req.encode("ascii"))
        resp = b""
        while b"\r\n\r\n" not in resp:
            resp += sock.recv(4096)
        body = {"id": msg_id, "method": method}
        if params:
            body["params"] = params
        _ws_send(sock, json.dumps(body))
        while True:
            raw = _ws_recv(sock)
            msg = json.loads(raw)
            if msg.get("id") == msg_id:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})
    finally:
        sock.close()


def export_pdf(html_path: Path, pdf_path: Path, chrome: str = "google-chrome") -> None:
    html_path = html_path.resolve()
    pdf_path = pdf_path.resolve()
    url = html_path.as_uri()

    proc = subprocess.Popen(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--remote-debugging-port=0",
            "about:blank",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    port = None
    deadline = time.time() + 30
    while time.time() < deadline:
        line = proc.stderr.readline()
        if not line:
            if proc.poll() is not None:
                break
            time.sleep(0.05)
            continue
        marker = "DevTools listening on ws://127.0.0.1:"
        if marker in line:
            # e.g. DevTools listening on ws://127.0.0.1:9222/devtools/browser/...
            after = line.strip().split(marker, 1)[1]
            port = int(after.split("/", 1)[0])
            break
    if port is None:
        proc.kill()
        raise RuntimeError(f"Could not read Chrome debug port. stderr: {proc.stderr.read()}")

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=10) as resp:
            version = json.load(resp)
        ws_url = version["webSocketDebuggerUrl"]

        _cdp_call(ws_url, "Target.createTarget", {"url": url}, msg_id=1)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=10) as resp:
            targets = json.load(resp)
        page = next(t for t in targets if t.get("type") == "page" and t.get("url", "").startswith("file:"))
        page_ws = page["webSocketDebuggerUrl"]

        _cdp_call(page_ws, "Page.enable", {}, msg_id=2)
        time.sleep(1.5)
        result = _cdp_call(
            page_ws,
            "Page.printToPDF",
            {
                "landscape": True,
                "paperWidth": PAPER_WIDTH_IN,
                "paperHeight": PAPER_HEIGHT_IN,
                "marginTop": 0,
                "marginBottom": 0,
                "marginLeft": 0,
                "marginRight": 0,
                "printBackground": True,
                "preferCSSPageSize": True,
                "scale": 1,
            },
            msg_id=3,
        )
        import base64

        pdf_path.write_bytes(base64.b64decode(result["data"]))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    out_dir = Path(__file__).resolve().parent
    targets = [
        (out_dir / "poster.html", out_dir / "poster.pdf"),
        (out_dir / "zh" / "poster.html", out_dir / "zh" / "poster.pdf"),
    ]
    if len(sys.argv) == 3:
        targets = [(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())]

    for html, pdf in targets:
        if not html.is_file():
            print(f"skip (missing): {html}", file=sys.stderr)
            continue
        print(f"Exporting {html.name} -> {pdf}")
        export_pdf(html, pdf)
        print(f"  wrote {pdf} ({pdf.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

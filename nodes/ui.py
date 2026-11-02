"""dora node: tiny web UI for the interactive demo (stdlib only).

  http://127.0.0.1:8765  -> live simulation view (MJPEG) + instruction box (+ voice input)

inputs   live_frame (JPEG bytes from sim)
outputs  command    JSON string {"text": "..."} or {"reset": seed}
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pyarrow as pa
from dora import Node

PORT = int(os.environ.get("UI_PORT", "8765"))
latest: dict[str, bytes] = {"frame": b""}
commands: queue.Queue[dict] = queue.Queue()
history: list[str] = []

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Jev tabletop</title>
<style>
body{background:#14171e;color:#e6e6e6;font-family:-apple-system,Helvetica,sans-serif;margin:16px}
img{width:100%;max-width:1280px;border-radius:6px}
.row{display:flex;gap:8px;margin-top:10px;max-width:1280px}
input{flex:1;font-size:18px;padding:10px;border-radius:6px;border:1px solid #444;background:#1e222b;color:#fff}
button{font-size:16px;padding:10px 16px;border-radius:6px;border:0;background:#f2c14e;color:#111;cursor:pointer}
button.alt{background:#3a4152;color:#eee}
#hist{margin-top:10px;color:#9aa6bf;font-size:14px}
</style></head><body>
<img src="/stream">
<div class="row">
 <input id="t" placeholder="e.g. Put the cubes with the colours of a traffic light into the tray." autofocus>
 <button onclick="send()">Send</button>
 <button class="alt" id="mic" onclick="listen()">🎤 Speak</button>
 <select id="lang"><option value="en-US">EN</option><option value="ja-JP">JA</option></select>
 <button class="alt" onclick="reset()">Shuffle cubes</button>
</div>
<div id="hist"></div>
<script>
const t=document.getElementById('t');
t.addEventListener('keydown',e=>{if(e.key==='Enter')send()});
async function post(o){await fetch('/command',{method:'POST',body:JSON.stringify(o)});refresh()}
function send(){if(t.value.trim()){post({text:t.value});t.value=''}}
function reset(){post({reset:Math.floor(Math.random()*1000)})}
async function refresh(){const h=await (await fetch('/history')).json();
 document.getElementById('hist').innerHTML=h.map(x=>'› '+x).reverse().join('<br>')}
function listen(){const R=window.SpeechRecognition||window.webkitSpeechRecognition;
 if(!R){alert('Speech recognition not supported in this browser');return}
 const r=new R();r.lang=document.getElementById('lang').value;r.interimResults=true;
 document.getElementById('mic').textContent='🎤 listening…';
 r.onresult=e=>{t.value=Array.from(e.results).map(x=>x[0].transcript).join('');
   if(e.results[e.results.length-1].isFinal){send()}};
 r.onend=()=>{document.getElementById('mic').textContent='🎤 Speak'};r.start()}
refresh();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A002  silence access log
        pass

    def do_GET(self) -> None:
        if self.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/history":
            body = json.dumps(history[-10:], ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = b""
            try:
                while True:
                    f = latest["frame"]
                    if f and f is not last:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(f)}\r\n\r\n".encode() + f + b"\r\n")
                        last = f
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/command":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", "0"))
        cmd = json.loads(self.rfile.read(n) or b"{}")
        if cmd.get("text"):
            history.append(cmd["text"])
        commands.put(cmd)
        self.send_response(204)
        self.end_headers()


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"UI on http://127.0.0.1:{PORT}", flush=True)
    node = Node()
    while True:
        ev = node.next(timeout=0.1)
        while not commands.empty():
            node.send_output("command", pa.array([json.dumps(commands.get(), ensure_ascii=False)]))
        if ev is None:
            continue
        if ev["type"] == "STOP":
            break
        if ev["type"] == "INPUT" and ev["id"] == "live_frame":
            latest["frame"] = ev["value"].to_numpy().tobytes()
    server.shutdown()


if __name__ == "__main__":
    main()

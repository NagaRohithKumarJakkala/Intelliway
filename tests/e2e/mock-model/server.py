import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

request_count = 0
fail_requests = False
state_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        global fail_requests
        if self.path == "/__stats":
            body = json.dumps({"requests": request_count}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/__fail":
            fail_requests = True
            self.send_response(204)
            self.end_headers()
            return
        if self.path == "/__recover":
            fail_requests = False
            self.send_response(204)
            self.end_headers()
            return
        self.send_error(404)

    def do_POST(self):
        global request_count
        length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        with state_lock:
            request_count += 1
            should_fail = fail_requests
        if should_fail:
            self.send_error(503, "controlled e2e failure")
            return
        if self.path.endswith("/embeddings"):
            payload = {
                "object": "list",
                "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
                "model": request.get("model", "e2e-mock"),
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        else:
            payload = {
                "id": "e2e-chat-1",
                "object": "chat.completion",
                "created": 1,
                "model": request.get("model", "e2e-mock"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "e2e response"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }
        if request.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = {
                "id": "e2e-stream-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": request.get("model", "e2e-mock"),
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "e2e stream"}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())
            return
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()

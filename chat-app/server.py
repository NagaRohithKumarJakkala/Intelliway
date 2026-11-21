import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).parent
STATIC = ROOT / "static"
PORT = int(os.getenv("PORT", "3000"))
LITELLM_URL = os.getenv("LITELLM_BASE_URL", "http://host.docker.internal:4000").rstrip("/")
ROUTER_URL = os.getenv(
    "SEMANTIC_ROUTER_BASE_URL",
    "http://host.docker.internal:8899",
).rstrip("/")
MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "")
DEFAULT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "4096"))

OPTIONS = [
    {"id": "auto", "label": "Auto (Semantic Router)", "endpoint": "router"},
    {"id": "auto-local", "label": "Auto Local", "endpoint": "router"},
    {"id": "auto-balanced", "label": "Auto Balanced", "endpoint": "router"},
    {"id": "auto-quality", "label": "Auto Quality", "endpoint": "router"},
    {"id": "gemma4", "label": "Gemma 4 (Local)", "endpoint": "litellm"},
    {"id": "ornith", "label": "Ornith (Local)", "endpoint": "litellm"},
    {"id": "glm-5.3", "label": "GLM 5.3", "endpoint": "litellm"},
    {"id": "gpt-6.1-terra", "label": "GPT 6.1 Terra", "endpoint": "litellm"},
]


def configured_options():
    options = list(OPTIONS)
    if not MASTER_KEY:
        return options
    try:
        upstream = Request(
            f"{LITELLM_URL}/v1/models",
            headers={"Authorization": f"Bearer {MASTER_KEY}"},
            method="GET",
        )
        with urlopen(upstream, timeout=10) as response:
            payload = json.loads(response.read())
        known = {item["id"] for item in options}
        for model in payload.get("data", []):
            model_id = model.get("id")
            if model_id and model_id not in known:
                options.append({
                    "id": model_id,
                    "label": f"{model_id} (LiteLLM)",
                    "endpoint": "litellm",
                })
                known.add(model_id)
    except Exception:
        pass
    return options


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/options":
            self.send_json(200, {"options": configured_options()})
            return
        if self.path in ("/", "/index.html"):
            self.serve_file(STATIC / "index.html", "text/html; charset=utf-8")
            return
        self.send_error(404)

    def serve_file(self, path, content_type):
        try:
            body = path.read_bytes()
        except FileNotFoundError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            model = request.get("model", "auto")
            messages = request.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a non-empty list")
            option = next((item for item in configured_options() if item["id"] == model), None)
            if option is None:
                raise ValueError("unknown model option")

            base_url = ROUTER_URL if option["endpoint"] == "router" else LITELLM_URL
            payload = {
                "model": model,
                "messages": messages,
                "temperature": request.get("temperature", 0.7),
                "max_tokens": request.get("max_tokens", DEFAULT_MAX_TOKENS),
            }
            headers = {"Content-Type": "application/json"}
            if MASTER_KEY:
                headers["Authorization"] = f"Bearer {MASTER_KEY}"

            upstream = Request(
                f"{base_url}/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            )
            with urlopen(upstream, timeout=180) as response:
                result = json.loads(response.read())
            self.send_json(200, result)
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json(400, {"error": str(error)})
        except HTTPError as error:
            detail = error.read().decode(errors="replace")
            self.send_json(error.code, {"error": detail or error.reason})
        except URLError as error:
            self.send_json(502, {"error": f"Upstream unavailable: {error.reason}"})
        except TimeoutError:
            self.send_json(504, {"error": "Upstream request timed out"})
        except Exception as error:
            self.send_json(500, {"error": str(error)})


if __name__ == "__main__":
    print(f"Chat app listening on 0.0.0.0:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()

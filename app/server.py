"""Local dashboard: python3 app/server.py [port]  ->  http://localhost:8765 (serves app/ and results/ read-only)."""
import functools, http.server, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class H(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        path = path.split("?")[0]
        if path in ("/", "/index.html"):
            return os.path.join(ROOT, "app", "index.html")
        if path.startswith("/results/") and path.endswith(".json"):
            return os.path.join(ROOT, "results", os.path.basename(path))
        return os.path.join(ROOT, "app", "404")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"haltgate dashboard on http://localhost:{port}")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()

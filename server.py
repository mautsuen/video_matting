from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import mimetypes

# make sure .mjs served as JS
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")

class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Cross-origin isolation (needed for wasm threads / SharedArrayBuffer)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
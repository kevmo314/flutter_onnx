"""Serve a local model comparison to an isolated, headless Chromium process."""

import json
import mimetypes
import queue
import re
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class DeviceRunner:
    def __init__(self, folder, javascript, chrome, device=None, timeout=900):
        self.folder = folder
        self.timeout = timeout
        self.requests = queue.Queue()
        self.replies = queue.Queue()
        self.files = {"/client.js": javascript} if chrome else {}
        self.last_route = ""
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def send(self, size, mime):
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(size))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()

            def do_GET(self):
                bridge.last_route = "GET " + self.path
                if self.path == "/":
                    body = b'<!doctype html><meta charset=utf-8><script defer src="/client.js"></script>'
                elif self.path == "/next":
                    body = json.dumps(bridge.requests.get()).encode()
                elif self.path in bridge.files:
                    path = bridge.files[self.path]
                    self.send(
                        path.stat().st_size,
                        mimetypes.guess_type(str(path))[0]
                        or "application/octet-stream",
                    )
                    with path.open("rb") as file:
                        shutil.copyfileobj(file, self.wfile, 1024 * 1024)
                    return
                else:
                    self.send_error(404)
                    return
                self.send(
                    len(body), "text/html" if self.path == "/" else "application/json"
                )
                self.wfile.write(body)

            def do_POST(self):
                bridge.last_route = "POST " + self.path
                length = int(self.headers["Content-Length"])
                if self.path == "/reply":
                    bridge.replies.put(json.loads(self.rfile.read(length)))
                elif re.fullmatch(r"/output/[0-9]+", self.path):
                    (
                        bridge.folder / ("browser-" + self.path.split("/")[-1] + ".bin")
                    ).write_bytes(self.rfile.read(length))
                else:
                    self.send_error(404)
                    return
                self.send(0, "text/plain")

        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 17889 if device else 0), Handler
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        address = f"http://127.0.0.1:{self.server.server_port}/"
        self.log = (folder / "chromium.log").open("w+")
        self.chrome = None
        if device == "android":
            package = "com.example.flutter_onnx_example"
            subprocess.run(["adb", "reverse", "tcp:17889", "tcp:17889"], check=True)
            subprocess.run(["adb", "shell", "am", "force-stop", package], check=True)
            subprocess.run(
                ["adb", "shell", "am", "start", "-n", package + "/.MainActivity"],
                check=True,
            )
            return
        if device:
            bundle = "com.example.flutterOnnxExample"
            subprocess.run(
                ["xcrun", "simctl", "terminate", device, bundle], check=False
            )
            subprocess.run(["xcrun", "simctl", "launch", device, bundle], check=True)
            return
        self.chrome = subprocess.Popen(
            [
                chrome,
                "--headless",
                "--enable-logging=stderr",
                "--disable-background-networking",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                f"--user-data-dir={folder / 'chrome-profile'}",
                address,
            ],
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )

    def run(self, request):
        self.files["/model"] = Path(request["model"])

        def feed(spec, label):
            if spec is None or isinstance(spec, str):
                return spec
            if isinstance(spec, list):
                return [feed(item, f"{label}-{i}") for i, item in enumerate(spec)]
            route = f"/feed/{label}"
            self.files[route] = Path(spec["file"])
            return {**spec, "file": route}

        feeds = {
            name: feed(spec, i)
            for i, (name, spec) in enumerate(request["feeds"].items())
        }
        external = {}
        for index, (location, path) in enumerate(
            request.get("externalData", {}).items()
        ):
            route = f"/sidecar/{index}"
            self.files[route] = Path(path)
            external[location] = route
        self.requests.put({**request, "feeds": feeds, "externalData": external})
        try:
            reply = self.replies.get(timeout=self.timeout)
        except queue.Empty:
            self.log.seek(0)
            raise TimeoutError(
                f"Device timed out on {request['model']} after {self.last_route}: "
                + self.log.read()[-1600:]
            ) from None

        def unpack(spec):
            if spec is None:
                return
            if isinstance(spec, list):
                for item in spec:
                    unpack(item)
                return
            spec["file"] = str(
                self.folder / ("browser-" + spec["file"].split("/")[-1] + ".bin")
            )

        for spec in reply.get("outputs", {}).values():
            unpack(spec)
        return reply

    def close(self):
        self.requests.put(None)
        if self.chrome is not None:
            self.chrome.terminate()
            self.chrome.wait(timeout=30)
        self.server.shutdown()
        self.server.server_close()
        self.log.close()

"""Exercise the real Windows downloader using tiny local HTTP fixtures."""

import hashlib
import http.server
import json
import os
import pathlib
import subprocess
import tempfile
import threading
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAYLOAD = b"Gyro downloader fixture: unchanged bytes.\n" * 16
FILENAME = "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf"


class Weights(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        offset = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
        self.send_response(206 if offset else 200)
        self.send_header("Content-Length", str(len(PAYLOAD) - offset))
        if offset:
            self.send_header(
                "Content-Range", f"bytes {offset}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}"
            )
        self.end_headers()
        self.wfile.write(PAYLOAD[offset:])

    def log_message(self, *args):
        pass


@unittest.skipUnless(os.name == "nt", "Windows PowerShell downloader")
class Downloader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Weights)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def exercise(self, kind):
        with tempfile.TemporaryDirectory(prefix="gyro & ! 50% ") as temp:
            root = pathlib.Path(temp)
            target = root / "Model Folder"
            target.mkdir()
            model, partial = target / FILENAME, target / (FILENAME + ".download")
            pin = dict(
                models=dict(
                    repo="fixture/model",
                    revision="0" * 40,
                    files=[
                        dict(
                            rfilename=FILENAME,
                            size=len(PAYLOAD),
                            lfs=dict(sha256=hashlib.sha256(PAYLOAD).hexdigest()),
                        )
                    ],
                )
            )
            (root / "pins.json").write_text(json.dumps(pin))
            script = (ROOT / "Download-Gyro-M.ps1").read_text(encoding="utf-8-sig")
            script = script.replace(
                '$Url = "https://huggingface.co/$Repository/resolve/$Revision/$FileName?download=true"',
                f'$Url = "http://127.0.0.1:{self.server.server_port}/weights"',
            )
            script = script.replace(
                "& $Curl --fail", "& $Curl --silent --show-error --fail"
            )
            path = root / "Download-Gyro-M.ps1"
            path.write_text(script, encoding="utf-8")
            if kind == "existing":
                model.write_bytes(PAYLOAD)
            elif kind == "resume":
                partial.write_bytes(PAYLOAD[:100])
            elif kind == "completed":
                partial.write_bytes(PAYLOAD)
            elif kind == "wrong-size":
                model.write_bytes(b"preserve me")
            elif kind == "wrong-hash":
                partial.write_bytes(b"X" * len(PAYLOAD))
            result = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(path),
                    "-Destination",
                    str(target),
                ],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            if kind in ("wrong-size", "wrong-hash"):
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                if kind == "wrong-size":
                    self.assertEqual(model.read_bytes(), b"preserve me")
                else:
                    self.assertEqual(partial.read_bytes(), b"X" * len(PAYLOAD))
                    self.assertFalse(model.exists())
                self.assertFalse((root / ".model-dir").exists())
            else:
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(model.read_bytes(), PAYLOAD)
                self.assertFalse(partial.exists())
                saved = pathlib.Path(
                    (root / ".model-dir").read_text(encoding="utf-8-sig")
                )
                self.assertTrue(saved.samefile(target))

    def test_fresh_download(self):
        self.exercise("fresh")

    def test_partial_download_resumes(self):
        self.exercise("resume")

    def test_completed_partial_is_verified_without_redownload(self):
        self.exercise("completed")

    def test_existing_model_is_reused(self):
        self.exercise("existing")

    def test_wrong_existing_model_is_preserved(self):
        self.exercise("wrong-size")

    def test_bad_partial_is_preserved(self):
        self.exercise("wrong-hash")


if __name__ == "__main__":
    unittest.main()

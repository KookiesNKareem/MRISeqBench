"""Cloud-host mailbox courier, outside the offline author sandbox."""

import json
import subprocess
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    directory = Path(spec["directory"])
    stop = threading.Event()
    seen = set()

    def courier(path):
        try:
            request = urllib.request.Request(
                spec.get("broker_url", "http://127.0.0.1:8766")
                + "/"
                + spec["token"]
                + "/"
                + path.stem,
                data=path.read_bytes(),
                headers={"Content-Type": "application/json"},
            )
            with (
                urllib.request.urlopen(request, timeout=86400) as response,
                path.with_suffix(".response").open("wb") as out,
            ):
                while True:
                    line = response.readline()
                    if not line:
                        break
                    out.write(line)
                    out.flush()
                    if json.loads(line).get("done"):
                        break
        except (OSError, ValueError, TypeError) as exc:
            path.with_suffix(".response").write_text(
                json.dumps(
                    {"error": "remote courier failure: " + str(exc), "done": True}
                )
                + "\n"
            )

    with ThreadPoolExecutor(max_workers=4) as pool:

        def watch():
            while not stop.wait(0.02):
                for path in directory.glob("*.request"):
                    if path.name not in seen:
                        seen.add(path.name)
                        pool.submit(courier, path)

        thread = threading.Thread(target=watch, daemon=True)
        thread.start()
        process = subprocess.Popen(spec["command"], stdin=subprocess.DEVNULL)
        try:
            return process.wait()
        finally:
            stop.set()
            thread.join(2)


if __name__ == "__main__":
    raise SystemExit(main())

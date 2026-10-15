"""Open the dashboard in its own window, running the server only while it's open.

    python -m dashboard.desktop

Starts Streamlit on a free local port (reachable only from this computer),
waits until it answers, shows it in a native window (WebKit, via pywebview),
and stops the server as soon as the window is closed. Needs the optional
extra: ``pip install -e ".[app]"``. The data folder comes from
``GARMINXSLEEP_DATA_DIR`` as usual.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

APP_SCRIPT = Path(__file__).resolve().parent / "app.py"
WINDOW_TITLE = "GarminxSleep"
STARTUP_TIMEOUT_SECONDS = 90


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def start_server(port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP_SCRIPT),
            "--server.headless=true",
            "--server.address=127.0.0.1",
            f"--server.port={port}",
            "--browser.gatherUsageStats=false",
        ],
        cwd=APP_SCRIPT.parent.parent,
    )


def wait_until_ready(server: subprocess.Popen, port: int) -> None:
    """Block until Streamlit's health check answers, or fail clearly."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    url = f"http://127.0.0.1:{port}/_stcore/health"
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"The dashboard server exited (code {server.returncode}).")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.25)
    raise RuntimeError("The dashboard server did not start in time.")


def stop_server(server: subprocess.Popen) -> None:
    if server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


def _name_application() -> None:
    """Show "GarminxSleep" instead of "Python" in the macOS menu bar."""
    try:
        from Foundation import NSBundle

        info = NSBundle.mainBundle().localizedInfoDictionary() or NSBundle.mainBundle().infoDictionary()
        if info is not None:
            info["CFBundleName"] = WINDOW_TITLE
    except Exception:
        pass  # Cosmetic only; other platforms or a missing PyObjC are fine.


def main() -> int:
    import webview

    port = _free_port()
    server = start_server(port)
    try:
        wait_until_ready(server, port)
        _name_application()
        webview.create_window(
            WINDOW_TITLE,
            f"http://127.0.0.1:{port}",
            width=1440,
            height=900,
            min_size=(900, 600),
        )
        webview.start()  # Returns when the window is closed.
    finally:
        stop_server(server)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

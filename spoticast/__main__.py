import threading
import time
import webbrowser

import uvicorn

from spoticast.config import settings


def main():
    host = "127.0.0.1"
    port = settings.port

    def open_browser():
        time.sleep(1.0)
        webbrowser.open(f"http://{host}:{port}")

    t = threading.Thread(target=open_browser, daemon=True)
    t.start()

    uvicorn.run(
        "spoticast.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()

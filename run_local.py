"""Start LungLens on this computer and open it in the browser (used by Start LungLens.bat)."""
import os
import socket
import sys
import threading
import traceback
import webbrowser


def free_port(preferred=5000):
    """Use 5000 if it is free, otherwise any free port (e.g. another copy of the app is already running)."""
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue


def main():
    from waitress import serve

    print("Loading the models, please wait...", flush=True)
    import app  # loads both models before the browser opens

    status = app.model_status()
    for name in ("cnn", "clinical"):
        ok = status[name]["ready"]
        print(f"  {name:9s}: {'OK' if ok else 'ERROR - ' + str(status[name]['error'])}")

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"\nLungLens is running at {url}")
    print("Keep this window open during the demo. Close it (or press Ctrl+C) to stop.\n", flush=True)
    if not os.getenv("NO_BROWSER"):
        threading.Timer(1.0, webbrowser.open, args=[url]).start()
    serve(app.app, host="127.0.0.1", port=port, threads=4)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("\nSomething went wrong. Take a screenshot of this window.")
        sys.exit(1)

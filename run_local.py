"""Start LungLens on this computer and open it in the browser (used by Start LungLens.bat)."""
import os
import threading
import webbrowser

from waitress import serve

print("Loading the models, please wait...", flush=True)
import app  # noqa: E402  (loads both models before the browser opens)

URL = "http://127.0.0.1:5000"
status = app.model_status()
for name in ("cnn", "clinical"):
    ok = status[name]["ready"]
    print(f"  {name:9s}: {'OK' if ok else 'ERROR - ' + str(status[name]['error'])}")

print(f"\nLungLens is running at {URL}")
print("Keep this window open during the demo. Close it (or press Ctrl+C) to stop.\n", flush=True)
if not os.getenv("NO_BROWSER"):
    threading.Timer(1.0, webbrowser.open, args=[URL]).start()
serve(app.app, host="127.0.0.1", port=5000, threads=4)

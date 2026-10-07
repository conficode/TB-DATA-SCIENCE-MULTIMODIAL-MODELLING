"""Make sure the real model files are present before the app starts.

The models are stored with Git LFS. Most hosts (Railway, Render, ...) clone the repo
WITHOUT fetching LFS objects, so the files on the server are tiny text pointers.
This script detects missing/pointer files and downloads the real ones from GitHub.

Override the source with MODEL_BASE_URL (or CLINICAL_MODEL_URL).
"""
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

BASE_URL = os.getenv(
    "MODEL_BASE_URL",
    "https://media.githubusercontent.com/media/conficode/TB-DATA-SCIENCE-MULTIMODIAL-MODELLING/main",
)
# The CNN runs from models/cnn/tb_xray_weights.npz (a regular git file), so only the LFS clinical model is fetched.
FILES = [
    (C.CLINICAL_MODEL_PATH, os.getenv("CLINICAL_MODEL_URL", f"{BASE_URL}/models/structured/tb_clinical_model.joblib")),
]


def needs_download(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return True
    with open(path, "rb") as f:
        return f.read(40).startswith(b"version https://git-lfs")


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"[fetch_models] downloading {url} -> {path}", flush=True)
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as out:
        while chunk := r.read(1 << 20):
            out.write(chunk)
    if needs_download(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{url} returned an LFS pointer or empty file")
    tmp.replace(path)
    print(f"[fetch_models] ok ({path.stat().st_size / 1e6:.1f} MB)", flush=True)


def main() -> None:
    for path, url in FILES:
        if needs_download(path):
            download(url, path)
        else:
            print(f"[fetch_models] {path.name} already present", flush=True)


if __name__ == "__main__":
    main()

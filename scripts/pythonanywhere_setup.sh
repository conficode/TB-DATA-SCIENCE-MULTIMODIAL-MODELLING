#!/bin/bash
# One-shot setup for a FREE PythonAnywhere account (512 MB disk, no TensorFlow needed).
#
# 1. Web tab -> "Add a new web app" -> Manual configuration -> Python 3.11
# 2. Open a Bash console and run:
#      curl -sL https://raw.githubusercontent.com/conficode/TB-DATA-SCIENCE-MULTIMODIAL-MODELLING/main/scripts/pythonanywhere_setup.sh | bash
# 3. Web tab -> Virtualenv: /home/<you>/.virtualenvs/lunglens -> Reload
#
# Free accounts can't download Git LFS files, so tb_xray_best.keras stays a pointer and the app loads
# models/cnn/tb_xray_weights.npz instead: the same weights, exported from the .keras file and verified
# against Keras.
set -euo pipefail

REPO=https://github.com/conficode/TB-DATA-SCIENCE-MULTIMODIAL-MODELLING.git
APP_DIR="$HOME/lunglens"
VENV="$HOME/.virtualenvs/lunglens"
PY=python3.11
WSGI="/var/www/${USER}_pythonanywhere_com_wsgi.py"

echo "==> Getting the code"
if [ -d "$APP_DIR/.git" ]; then
    git -C "$APP_DIR" pull --ff-only
else
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "$REPO" "$APP_DIR"
fi

echo "==> Creating the virtualenv (reuses PythonAnywhere's preinstalled packages to save disk)"
[ -d "$VENV" ] || $PY -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install --no-cache-dir -q -r "$APP_DIR/requirements.txt"

echo "==> Checking that both models load"
cd "$APP_DIR"
"$VENV/bin/python" -c "import app, json; s = app.model_status(); print(json.dumps(s, indent=1)); assert s['cnn']['ready'] and s['clinical']['ready']"

echo "==> Writing the WSGI file"
if [ ! -f "$WSGI" ]; then
    echo "!! $WSGI not found. Create the web app first (Web tab -> Add a new web app -> Manual configuration -> Python 3.11), then rerun this script."
    exit 1
fi
cat > "$WSGI" <<EOF
import os, sys
path = "$APP_DIR"
if path not in sys.path:
    sys.path.insert(0, path)
os.chdir(path)
os.environ.setdefault("SECRET_KEY", "$(head -c 24 /dev/urandom | base64 | tr -d '/+=')")
from app import app as application
EOF

echo
echo "Done. In the Web tab:"
echo "  1. Virtualenv: $VENV"
echo "  2. Click the green Reload button"
echo "  3. Open https://${USER}.pythonanywhere.com/health"
du -sh "$HOME" 2>/dev/null | sed 's/^/Disk used: /'

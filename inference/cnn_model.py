"""CNN inference: exact training preprocessing + Grad-CAM (logic from the xray notebook).

The network (Rescaling -> 3x [Conv3x3+ReLU, MaxPool2] -> Flatten -> Dense128+ReLU -> Dense1+sigmoid)
is evaluated in pure NumPy. The weights are read straight from tb_xray_best.keras (a zip holding
model.weights.h5) with h5py, so TensorFlow (~1.5 GB RAM) is not needed at runtime and the app fits
512 MB hosts. Outputs match Keras to float32 precision (verified by scripts/export_cnn_weights.py --verify).
If the .keras file is unavailable, the NumPy export tb_xray_weights.npz is used instead.
"""
import logging
import re
import tempfile
import zipfile
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
import config as C

log = logging.getLogger(__name__)


def _conv_relu(x, w, b):
    """'valid' 3x3 conv, stride 1, channels_last. x: (H, W, Cin), w: (3, 3, Cin, Cout)."""
    win = sliding_window_view(x, w.shape[:2], axis=(0, 1))               # (H-2, W-2, Cin, 3, 3)
    out = np.einsum("hwcij,ijco->hwo", win, w, optimize=True) + b
    return np.maximum(out, 0, out=out)


def _maxpool(x):
    """2x2 / stride 2 'valid' max pool. Returns pooled map and argmax (0..3, row-major, first max wins)."""
    h, w, c = x.shape[0] // 2, x.shape[1] // 2, x.shape[2]
    blocks = x[:h * 2, :w * 2].reshape(h, 2, w, 2, c).transpose(0, 2, 4, 1, 3).reshape(h, w, c, 4)
    idx = blocks.argmax(-1)
    return np.take_along_axis(blocks, idx[..., None], -1)[..., 0], idx


def _maxpool_backward(grad, idx, shape):
    h, w, c = grad.shape
    blocks = np.zeros((h, w, c, 4), dtype=grad.dtype)
    np.put_along_axis(blocks, idx[..., None], grad[..., None], -1)
    out = np.zeros(shape, dtype=grad.dtype)
    out[:h * 2, :w * 2] = blocks.reshape(h, w, c, 2, 2).transpose(0, 3, 1, 4, 2).reshape(h * 2, w * 2, c)
    return out


# matplotlib's 'jet' colormap (same segment data, 256-entry lookup table) without the dependency
_JET_SEGMENTS = [((0, .35, .66, .89, 1), (0, 0, 1, 1, .5)),
                 ((0, .125, .375, .64, .91, 1), (0, 0, 1, 1, 0, 0)),
                 ((0, .11, .34, .65, 1), (.5, 1, 1, 0, 0))]
_JET_LUT = np.stack([np.interp(np.linspace(0, 1, 256), x, y) for x, y in _JET_SEGMENTS], axis=-1)


def _jet(values):
    """values in [0, 1] -> RGB floats in [0, 1], identical to matplotlib.cm.jet(values)[..., :3]."""
    return _JET_LUT[np.clip((values * 256).astype(int), 0, 255)]


def _resize_bilinear(img, size):
    """tf.image.resize(method='bilinear', antialias=False): half-pixel centres, edge clamping."""
    out_h, out_w = size

    def axis(n_in, n_out):
        src = (np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5
        lo = np.floor(src)
        frac = (src - lo).astype(np.float32)
        return np.clip(lo, 0, n_in - 1).astype(int), np.clip(lo + 1, 0, n_in - 1).astype(int), frac

    y0, y1, fy = axis(img.shape[0], out_h)
    x0, x1, fx = axis(img.shape[1], out_w)
    top = img[y0][:, x0] + (img[y0][:, x1] - img[y0][:, x0]) * fx[None, :, None]
    bot = img[y1][:, x0] + (img[y1][:, x1] - img[y1][:, x0]) * fx[None, :, None]
    return top + (bot - top) * fy[:, None, None]


def _is_real_file(path):
    """False for a missing file or a Git LFS pointer (hosts that clone without LFS)."""
    if not path.exists() or path.stat().st_size < 1024:
        return False
    with open(path, "rb") as f:
        return not f.read(40).startswith(b"version https://git-lfs")


def load_keras_weights(path):
    """Read the Conv2D/Dense kernels and biases from a Keras-3 .keras file without TensorFlow."""
    import h5py
    with zipfile.ZipFile(path) as z, tempfile.TemporaryDirectory() as tmp:
        h5_path = z.extract("model.weights.h5", tmp)                 # extract to disk: keeps RAM low
        with h5py.File(h5_path, "r") as f:
            layers = {}
            for name, grp in f["layers"].items():
                if "vars" in grp and len(grp["vars"]) == 2:
                    layers[name] = (grp["vars"]["0"][()], grp["vars"]["1"][()])

    def ordered(prefix):        # Keras names layers conv2d, conv2d_1, conv2d_2 ... in model order
        names = [n for n in layers if re.fullmatch(rf"{prefix}(_\d+)?", n)]
        return sorted(names, key=lambda n: int(n.rsplit("_", 1)[1]) if n != prefix else 0)

    convs, denses = ordered("conv2d"), ordered("dense")
    if len(convs) != 3 or len(denses) != 2:
        raise ValueError(f"unexpected architecture in {path.name}: {sorted(layers)}")
    w = {}
    for i, n in enumerate(convs, 1):
        w[f"conv{i}_w"], w[f"conv{i}_b"] = layers[n]
    for i, n in enumerate(denses, 1):
        w[f"dense{i}_w"], w[f"dense{i}_b"] = layers[n]
    expected = {"conv1_w": (3, 3, 3, 32), "conv2_w": (3, 3, 32, 64), "conv3_w": (3, 3, 64, 128),
                "dense1_w": (26 * 26 * 128, 128), "dense2_w": (128, 1)}
    for k, shape in expected.items():
        if w[k].shape != shape:
            raise ValueError(f"{k} has shape {w[k].shape}, expected {shape}")
    return {k: v.astype(np.float32) for k, v in w.items()}


class CNNModel:
    def __init__(self, path=C.CNN_MODEL_PATH, fallback=C.CNN_WEIGHTS_PATH):
        self.w, self.error, self.path, self.source = None, None, path, None
        try:
            if _is_real_file(path):
                self.w, self.source = load_keras_weights(path), path.name
            elif fallback.exists():
                log.warning("%s missing or an LFS pointer; using %s", path.name, fallback.name)
                with np.load(fallback) as f:
                    self.w = {k: f[k].astype(np.float32) for k in f.files}
                self.source = fallback.name
            else:
                raise FileNotFoundError(f"CNN model not found at {path}. Copy tb_xray_best.keras into models/cnn/.")
            log.info("CNN loaded from %s", self.source)
        except Exception as e:                                        # app keeps running; UI shows the error
            self.error = str(e)
            log.error("CNN loading failed: %s", e)

    @property
    def ready(self):
        return self.w is not None

    def preprocess(self, image_path):
        """Same as training: decode as 3-channel RGB, bilinear resize to 224x224, float 0-255.
        (Rescaling 1/255 is the model's first layer and is applied in _forward.)"""
        with Image.open(image_path) as im:
            img = np.asarray(im.convert("RGB"), dtype=np.float32)
        return _resize_bilinear(img, C.CNN_IMG_SIZE)[None]

    def _forward(self, x):
        """Inference forward pass (augmentation and dropout inactive). Keeps what Grad-CAM needs."""
        w = self.w
        h = x[0] * np.float32(1 / 255)
        h, _ = _maxpool(_conv_relu(h, w["conv1_w"], w["conv1_b"]))
        h, _ = _maxpool(_conv_relu(h, w["conv2_w"], w["conv2_b"]))
        conv = _conv_relu(h, w["conv3_w"], w["conv3_b"])              # last Conv2D output (Grad-CAM target)
        pooled, idx = _maxpool(conv)
        z1 = pooled.reshape(-1) @ w["dense1_w"] + w["dense1_b"]
        hidden = np.maximum(z1, 0)
        p = 1 / (1 + np.exp(-(hidden @ w["dense2_w"] + w["dense2_b"])))
        return float(p[0]), (conv, pooled, idx, z1)

    def predict(self, image_path):
        x = self.preprocess(image_path)
        p, _ = self._forward(x)
        return p, x

    def gradcam(self, x, out_path, alpha=0.4):
        """Grad-CAM as in the notebook: gradients of P(TB) w.r.t. the last Conv2D feature map,
        channel weights = mean gradient. Backprop through the dense head is done analytically."""
        w = self.w
        p, (conv, pooled, idx, z1) = self._forward(x)
        d_z2 = p * (1 - p)
        d_z1 = d_z2 * w["dense2_w"][:, 0] * (z1 > 0)
        d_pooled = (w["dense1_w"] @ d_z1).reshape(pooled.shape)
        grads = _maxpool_backward(d_pooled, idx, conv.shape)
        weights = grads.mean(axis=(0, 1))
        cam = np.maximum((conv * weights).sum(-1), 0)
        cam = cam / cam.max() if cam.max() > 0 else cam
        heat = Image.fromarray(np.uint8(_jet(cam) * 255)).resize(C.CNN_IMG_SIZE, Image.BILINEAR)
        base = x[0]
        overlay = np.clip(np.asarray(heat, dtype=np.float32) * alpha + base, 0, 255).astype(np.uint8)
        Image.fromarray(overlay).save(out_path)
        Image.fromarray(base.astype(np.uint8)).save(str(out_path).replace("_cam.png", "_input.png"))
        return out_path


def validate_image(path):
    """Raise ValueError if the file is not a readable image."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            if min(im.size) < 64:
                raise ValueError("Image is too small to be a chest X-ray (minimum 64 px).")
    except ValueError:
        raise
    except Exception:
        raise ValueError("The uploaded file is not a valid image.")

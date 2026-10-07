"""CNN inference: exact training preprocessing + Grad-CAM (logic from the xray notebook).

The network (Rescaling -> 3x [Conv3x3+ReLU, MaxPool2] -> Flatten -> Dense128+ReLU -> Dense1+sigmoid)
is evaluated in pure NumPy from weights exported by scripts/export_cnn_weights.py. This avoids
TensorFlow at runtime (~1.5 GB RAM) so the app fits free 512 MB hosts. Outputs match Keras to
float32 precision (verified by scripts/export_cnn_weights.py --verify).
"""
import logging
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


class CNNModel:
    def __init__(self, path=C.CNN_WEIGHTS_PATH):
        self.w, self.error, self.path = None, None, path
        try:
            if not path.exists():
                raise FileNotFoundError(f"CNN weights not found at {path}. Run scripts/export_cnn_weights.py.")
            with np.load(path) as f:
                self.w = {k: f[k].astype(np.float32) for k in f.files}
            log.info("CNN loaded from %s", path.name)
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
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import cm
        heat = Image.fromarray(np.uint8(cm.jet(cam)[..., :3] * 255)).resize(C.CNN_IMG_SIZE, Image.BILINEAR)
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

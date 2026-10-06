"""CNN inference: exact training preprocessing + Grad-CAM (logic from the xray notebook)."""
import logging
import numpy as np
from PIL import Image
import config as C

log = logging.getLogger(__name__)


class CNNModel:
    def __init__(self, path=C.CNN_MODEL_PATH):
        self.model, self.error, self.path = None, None, path
        try:
            import tensorflow as tf
            self.tf = tf
            if not path.exists():
                raise FileNotFoundError(f"CNN model not found at {path}. Copy tb_xray_best.keras into models/cnn/.")
            self.model = tf.keras.models.load_model(path)
            self.model(tf.zeros((1, *C.CNN_IMG_SIZE, 3)), training=False)   # build graph once
            self.last_conv = next(l for l in reversed(self.model.layers) if isinstance(l, tf.keras.layers.Conv2D))
            log.info("CNN loaded (last conv layer: %s)", self.last_conv.name)
        except Exception as e:                                        # app keeps running; UI shows the error
            self.error = str(e)
            log.error("CNN loading failed: %s", e)

    @property
    def ready(self):
        return self.model is not None

    def preprocess(self, image_path):
        """Same as training: decode as 3-channel RGB, bilinear resize to 224x224, float 0-255.
        (Rescaling 1/255 is INSIDE the model, so it is not applied here.)"""
        tf = self.tf
        raw = tf.io.read_file(str(image_path))
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, C.CNN_IMG_SIZE, method="bilinear")
        return tf.expand_dims(tf.cast(img, tf.float32), 0)

    def predict(self, image_path):
        x = self.preprocess(image_path)
        p = float(self.model(x, training=False).numpy()[0][0])      # training=False -> augmentation inactive
        return p, x

    def gradcam(self, x, out_path, alpha=0.4):
        """Grad-CAM as in the notebook: forward pass from the Rescaling layer (skipping augmentation),
        gradients of P(TB) w.r.t. the last Conv2D feature map, channel weights = mean gradient."""
        tf = self.tf
        layers = self.model.layers
        start = next(i for i, l in enumerate(layers) if isinstance(l, tf.keras.layers.Rescaling))
        with tf.GradientTape() as tape:
            h, conv = x, None
            for layer in layers[start:]:
                h = layer(h, training=False)
                if layer.name == self.last_conv.name:
                    conv = h
                    tape.watch(conv)
            prob = h[:, 0]
        grads = tape.gradient(prob, conv)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
        cam = cam / cam.max() if cam.max() > 0 else cam
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import cm
        heat = Image.fromarray(np.uint8(cm.jet(cam)[..., :3] * 255)).resize(C.CNN_IMG_SIZE, Image.BILINEAR)
        base = x[0].numpy()
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

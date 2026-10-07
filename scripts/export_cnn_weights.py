"""Export tb_xray_best.keras to the NumPy weight file the app runs on (no TensorFlow needed at runtime).

Run once after retraining (needs TensorFlow locally):
    python scripts/export_cnn_weights.py            # writes models/cnn/tb_xray_weights.npz
    python scripts/export_cnn_weights.py --verify img1.png img2.jpg ...   # compare with Keras
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import config as C  # noqa: E402


def export():
    model = tf.keras.models.load_model(C.CNN_MODEL_PATH)
    convs = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    denses = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    assert len(convs) == 3 and len(denses) == 2, "architecture changed; update inference/cnn_model.py"
    arrays = {}
    for i, l in enumerate(convs, 1):
        arrays[f"conv{i}_w"], arrays[f"conv{i}_b"] = l.get_weights()
    for i, l in enumerate(denses, 1):
        arrays[f"dense{i}_w"], arrays[f"dense{i}_b"] = l.get_weights()
    np.savez_compressed(C.CNN_WEIGHTS_PATH, **{k: v.astype(np.float32) for k, v in arrays.items()})
    print(f"wrote {C.CNN_WEIGHTS_PATH} ({C.CNN_WEIGHTS_PATH.stat().st_size / 1e6:.1f} MB)")
    return model


def verify(model, images):
    from inference.cnn_model import CNNModel
    np_model = CNNModel()
    last_conv = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)][-1]
    start = next(i for i, l in enumerate(model.layers) if isinstance(l, tf.keras.layers.Rescaling))
    for path in images:
        raw = tf.io.read_file(str(path))
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        x_tf = tf.expand_dims(tf.cast(tf.image.resize(img, C.CNN_IMG_SIZE, method="bilinear"), tf.float32), 0)
        p_tf = float(model(x_tf, training=False).numpy()[0][0])
        with tf.GradientTape() as tape:
            h = x_tf
            for layer in model.layers[start:]:
                h = layer(h, training=False)
                if layer.name == last_conv.name:
                    conv = h
                    tape.watch(conv)
            prob = h[:, 0]
        g_tf = tape.gradient(prob, conv).numpy().mean(axis=(0, 1, 2))

        p_np, x_np = np_model.predict(path)
        _, (conv_np, pooled, idx, z1) = np_model._forward(x_np)
        w = np_model.w
        d_z1 = p_np * (1 - p_np) * w["dense2_w"][:, 0] * (z1 > 0)
        from inference.cnn_model import _maxpool_backward
        g_np = _maxpool_backward((w["dense1_w"] @ d_z1).reshape(pooled.shape), idx, conv_np.shape).mean(axis=(0, 1))
        print(f"{Path(path).name}: P(TB) keras={p_tf:.6f} numpy={p_np:.6f} | "
              f"max input diff={np.abs(x_tf.numpy() - x_np).max():.4f} | "
              f"grad-cam weight max diff={np.abs(g_tf - g_np).max():.2e} (scale {np.abs(g_tf).max():.2e})")


if __name__ == "__main__":
    m = export()
    if "--verify" in sys.argv:
        verify(m, sys.argv[sys.argv.index("--verify") + 1:])

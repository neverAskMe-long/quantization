"""
onnx_inference.py - Run every .onnx file in onnx_models/ on the CPU with ONNX Runtime,
measure accuracy + speed, and write results/onnx_benchmark.csv.
Run:  python onnx_inference.py
      python onnx_inference.py --image my_cat.png --onnx onnx_models/resnet18_fp32.onnx
"""
import argparse
import csv
import glob
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

from common import CLASSES, MEAN, STD, ONNX_DIR, RESULTS_DIR, get_test_loader


def make_session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1       # one thread = fair speed comparison
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def predict_image(session, image_path):
    img = Image.open(image_path).convert("RGB").resize((32, 32))
    x = (np.asarray(img, dtype=np.float32) / 255.0 - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    probs = softmax(session.run(None, {"image": x})[0])[0]
    return CLASSES[int(probs.argmax())], float(probs.max())


def benchmark_all():
    files = sorted(glob.glob(os.path.join(ONNX_DIR, "*.onnx")))
    if not files:
        print("No .onnx files found. Run export_onnx.py first.")
        return
    loader = get_test_loader(batch_size=200)
    batches = [(x.numpy(), y.numpy()) for x, y in loader]
    rows = []
    for path in files:
        name = os.path.basename(path)
        print(f"\nTesting {name} ...")
        try:
            sess = make_session(path)
            correct = total = 0
            for x, y in batches:
                out = sess.run(None, {"image": x})[0]
                correct += int((out.argmax(1) == y).sum())
                total += len(y)
            single = np.concatenate([b[0] for b in batches])[:220]
            for i in range(20):                                   # warm-up
                sess.run(None, {"image": single[i:i + 1]})
            times = []
            for i in range(20, 220):
                t0 = time.perf_counter()
                sess.run(None, {"image": single[i:i + 1]})
                times.append((time.perf_counter() - t0) * 1000)
            times = np.array(times)
            rows.append({"onnx_file": name, "accuracy_percent": round(100.0 * correct / total, 2),
                         "file_size_mb": round(os.path.getsize(path) / (1024 * 1024), 3),
                         "latency_ms_mean": round(times.mean(), 3),
                         "latency_ms_p50": round(float(np.percentile(times, 50)), 3),
                         "latency_ms_p99": round(float(np.percentile(times, 99)), 3)})
            print(f"  accuracy {rows[-1]['accuracy_percent']}%   latency {rows[-1]['latency_ms_mean']} ms/picture")
        except Exception as err:
            print(f"  FAILED: {type(err).__name__}: {str(err)[:300]}")

    if rows:
        out_path = os.path.join(RESULTS_DIR, "onnx_benchmark.csv")
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="classify one picture instead of benchmarking")
    parser.add_argument("--onnx", default=os.path.join(ONNX_DIR, "resnet18_fp32.onnx"))
    args = parser.parse_args()
    if args.image:
        label, conf = predict_image(make_session(args.onnx), args.image)
        print(f"Predicted: {label} ({conf * 100:.1f}%)")
    else:
        benchmark_all()


if __name__ == "__main__":
    main()
    

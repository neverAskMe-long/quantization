"""
benchmark.py - Tests the models on the full CIFAR-10 test set (10,000 pictures)
and measures accuracy, file size, number of weights and speed.
  python benchmark.py --group q1   -> fp32, ptq, qat
  python benchmark.py --group q2   -> fp32, unstructured, structured
  python benchmark.py              -> both
Output (results/): q1_predictions.csv, q1_benchmark.csv, q2_predictions.csv, q2_benchmark.csv
"""
import argparse
import csv
import os
import time
import zipfile

import numpy as np
import torch

from common import (CLASSES, Q1_MODELS, Q2_MODELS, RESULTS_DIR, count_macs, count_params,
                    file_size_mb, get_quant_backend, get_test_loader, load_model, model_path, set_seed)


@torch.no_grad()
def all_predictions(model, loader):
    """Run the whole test set. Returns labels, predictions, confidences."""
    labels, preds, confs = [], [], []
    for x, y in loader:
        probs = torch.softmax(model(x), dim=1)
        c, p = probs.max(dim=1)
        labels.append(y)
        preds.append(p)
        confs.append(c)
    return torch.cat(labels).numpy(), torch.cat(preds).numpy(), torch.cat(confs).numpy()


@torch.no_grad()
def measure_latency(model, loader, n=200, warmup=20):
    """Time ONE picture at a time (like a real device), in milliseconds."""
    images = torch.cat([x for x, _ in loader])[: n + warmup]
    for i in range(warmup):
        model(images[i:i + 1])
    times = []
    for i in range(warmup, warmup + n):
        t0 = time.perf_counter()
        model(images[i:i + 1])
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return times.mean(), np.percentile(times, 50), np.percentile(times, 99)


def zipped_size_mb(path):
    """Size after zip compression - shows the benefit of zeros in unstructured pruning."""
    tmp = path + ".tmp.zip"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path)
    size = os.path.getsize(tmp) / (1024 * 1024)
    os.remove(tmp)
    return size


def run_group(group_name, names):
    loader = get_test_loader(batch_size=200)
    pred_columns, rows = {}, []
    labels = None

    for name in names:
        path = model_path(name)
        if not os.path.exists(path):
            print(f"[{name}] model file missing, skipped")
            continue
        print(f"\nTesting {name} ...")
        model = load_model(name)
        labels, preds, confs = all_predictions(model, loader)
        accuracy = 100.0 * (preds == labels).mean()
        mean_ms, p50_ms, p99_ms = measure_latency(model, loader)

        if name in ("ptq", "qat"):
            total = nonzero = macs = sparsity = ""
        else:
            total, nonzero = count_params(model)
            macs = round(count_macs(model) / 1e6, 2)
            sparsity = round(100 * (1 - nonzero / total), 2)

        rows.append({
            "model": name, "accuracy_percent": round(accuracy, 2),
            "file_size_mb": round(file_size_mb(path), 3), "zipped_size_mb": round(zipped_size_mb(path), 3),
            "total_weights": total, "nonzero_weights": nonzero, "sparsity_percent": sparsity,
            "macs_millions": macs,
            "latency_ms_mean": round(mean_ms, 3), "latency_ms_p50": round(p50_ms, 3),
            "latency_ms_p99": round(p99_ms, 3),
            "throughput_img_per_s": round(1000.0 / mean_ms, 1),
        })
        pred_columns[name] = (preds, confs)
        print(f"  accuracy {accuracy:.2f}%   latency {mean_ms:.2f} ms/picture")

    if not rows:
        print(f"No models found for {group_name}.")
        return

    # per-picture predictions
    pred_path = os.path.join(RESULTS_DIR, f"{group_name}_predictions.csv")
    with open(pred_path, "w", newline="") as f:
        header = ["image_index", "true_label", "true_name"]
        for n in pred_columns:
            header += [f"{n}_pred", f"{n}_pred_name", f"{n}_confidence", f"{n}_correct"]
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(labels)):
            row = [i, int(labels[i]), CLASSES[labels[i]]]
            for n, (p, c) in pred_columns.items():
                row += [int(p[i]), CLASSES[p[i]], f"{c[i]:.4f}", int(p[i] == labels[i])]
            w.writerow(row)

    # summary
    bench_path = os.path.join(RESULTS_DIR, f"{group_name}_benchmark.csv")
    with open(bench_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved {pred_path}\nSaved {bench_path}")
    print(f"{'model':<14}{'acc %':>8}{'size MB':>10}{'ms/img':>9}")
    for r in rows:
        print(f"{r['model']:<14}{r['accuracy_percent']:>8}{r['file_size_mb']:>10}{r['latency_ms_mean']:>9}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["q1", "q2", "all"], default="all")
    args = parser.parse_args()

    set_seed(42)
    torch.set_num_threads(1)   # one CPU thread = fair, repeatable speed comparison
    torch.backends.quantized.engine = get_quant_backend()

    if args.group in ("q1", "all"):
        run_group("q1", Q1_MODELS)
    if args.group in ("q2", "all"):
        run_group("q2", Q2_MODELS)


if __name__ == "__main__":
    main()

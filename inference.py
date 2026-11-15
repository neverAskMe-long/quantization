"""
inference.py - Classify ONE picture with any of the five models.
  python inference.py --model fp32 --image cat.png
  python inference.py --model all  --image cat.png        (compare all five)
  python inference.py --model all  --test-index 7         (use CIFAR-10 test picture number 7)
Models: fp32, ptq, qat, unstructured, structured, all
"""
import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

from common import (CLASSES, MEAN, MODEL_FILES, STD, _cifar_arrays,
                    get_quant_backend, load_model, model_path)


def preprocess(image_path):
    """Open a picture, make it 32x32, scale to 0-1, normalise like training. Returns [1,3,32,32]."""
    img = Image.open(image_path).convert("RGB").resize((32, 32))
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (x - mean) / std


def load_test_picture(index):
    """Take picture number `index` from the CIFAR-10 test set. Returns (tensor, true class name)."""
    images, labels = _cifar_arrays(False)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    x = (images[index].float().div(255).unsqueeze(0) - mean) / std
    return x, CLASSES[int(labels[index])]


@torch.no_grad()
def predict(model, x, top_k=3):
    """Run the model once. Returns (top-k list of (class, percent), milliseconds)."""
    model(x)                                   # warm-up run (not timed)
    t0 = time.perf_counter()
    logits = model(x)
    ms = (time.perf_counter() - t0) * 1000
    probs = torch.softmax(logits, dim=1)[0]
    conf, idx = probs.topk(top_k)
    return [(CLASSES[int(i)], float(c) * 100) for c, i in zip(conf, idx)], ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=list(MODEL_FILES) + ["all"])
    parser.add_argument("--image", help="path to a picture (png/jpg)")
    parser.add_argument("--test-index", type=int, help="use this CIFAR-10 test picture instead of a file")
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.backends.quantized.engine = get_quant_backend()   # needed to run the INT8 models

    true_name = None
    if args.image:
        x = preprocess(args.image)
    elif args.test_index is not None:
        x, true_name = load_test_picture(args.test_index)
    else:
        parser.error("give --image PATH or --test-index N")

    names = list(MODEL_FILES) if args.model == "all" else [args.model]
    if true_name:
        print(f"True class: {true_name}")
    print(f"{'model':<14}{'prediction':<14}{'confidence':>11}{'ms':>9}   top-3")
    for name in names:
        if not os.path.exists(model_path(name)):
            print(f"{name:<14}model file missing, skipped")
            continue
        top, ms = predict(load_model(name), x)
        top3 = ", ".join(f"{c} {p:.0f}%" for c, p in top)
        print(f"{name:<14}{top[0][0]:<14}{top[0][1]:>10.1f}%{ms:>9.2f}   {top3}")


if __name__ == "__main__":
    main()

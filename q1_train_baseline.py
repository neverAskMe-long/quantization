"""
q1_train_baseline.py - STEP 1: train the normal (FP32) ResNet-18 on CIFAR-10 and save it.
Run:  python q1_train_baseline.py --epochs 30
"""
import argparse
import csv
import os
import time

import torch

from common import (RESULTS_DIR, build_model, evaluate, get_device, get_test_loader,
                    get_train_loader, model_path, set_seed, train_one_epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12, help="how many passes over the training pictures")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1, help="highest learning rate")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    print(f"Training on: {device}")

    train_loader = get_train_loader(args.batch_size)
    test_loader = get_test_loader()

    model = build_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    # learning rate goes up, then slowly down - gives good accuracy quickly
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))

    log_path = os.path.join(RESULTS_DIR, "train_log_fp32.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_acc", "seconds"])
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, scheduler)
            test_acc = evaluate(model, test_loader, device)
            secs = time.time() - start
            print(f"Epoch {epoch:02d}/{args.epochs}  loss {loss:.3f}  "
                  f"train acc {train_acc:.2f}%  test acc {test_acc:.2f}%  ({secs:.0f}s)")
            writer.writerow([epoch, f"{loss:.4f}", f"{train_acc:.2f}", f"{test_acc:.2f}", f"{secs:.1f}"])

    torch.save(model.cpu().state_dict(), model_path("fp32"))
    print(f"\nSaved FP32 model -> {model_path('fp32')}")
    print(f"Training log      -> {log_path}")


if __name__ == "__main__":
    main()

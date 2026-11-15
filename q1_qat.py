"""
q1_qat.py - Quantization-Aware Training (QAT).
The model PRACTISES being INT8 while it fine-tunes.
Run:  python q1_qat.py --epochs 8
"""
import argparse
import os

import torch
from torch.ao.nn.intrinsic.qat import freeze_bn_stats
from torch.ao.quantization import convert, disable_observer, get_default_qat_qconfig, prepare_qat

from common import (MODELS_DIR, build_model, evaluate, file_size_mb, get_device, get_quant_backend,
                    get_test_loader, get_train_loader, model_path, set_seed, state_dict_path, train_one_epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    train_loader = get_train_loader(args.batch_size)
    test_loader = get_test_loader()

    # 1. pretrained FP32 model
    model = build_model()
    model.load_state_dict(torch.load(model_path("fp32"), map_location="cpu"))

    # 2 + 3. fuse, set the QAT settings, insert fake-quant modules
    model.train()
    model.fuse_model(is_qat=True)
    backend = get_quant_backend()
    torch.backends.quantized.engine = backend
    model.qconfig = get_default_qat_qconfig(backend)
    prepare_qat(model, inplace=True)
    print(f"QAT prepared (backend: {backend}). Fine-tuning on: {device}")
    model.to(device)

    # 4. fine-tune
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    for epoch in range(1, args.epochs + 1):
        if epoch > max(1, args.epochs // 2):
            model.apply(freeze_bn_stats)   # stop BatchNorm from changing
        if epoch > max(2, (3 * args.epochs) // 4):
            model.apply(disable_observer)  # stop the number ranges from moving
        loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        print(f"QAT epoch {epoch}/{args.epochs}  loss {loss:.3f}  train acc {train_acc:.2f}%")

    # 5. convert to a real INT8 model (must be done on CPU)
    model.cpu().eval()
    int8_model = convert(model, inplace=False)
    int8_acc = evaluate(int8_model, test_loader)
    print(f"QAT INT8 accuracy: {int8_acc:.2f}%")

    # 6. save: TorchScript (runs anywhere) + state dict
    scripted = torch.jit.trace(int8_model, torch.randn(1, 3, 32, 32))
    torch.jit.save(scripted, model_path("qat"))
    torch.save(int8_model.state_dict(), state_dict_path("qat"))
    print(f"Saved INT8 model -> {model_path('qat')}  ({file_size_mb(model_path('qat')):.2f} MB)")

    # keep the fake-quant model too: export_onnx.py uses it later
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "resnet18_qat_fakequant_state_dict.pth"))


if __name__ == "__main__":
    main()

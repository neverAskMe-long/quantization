"""
q2_prune_structured.py - Structured CHANNEL pruning.
We delete whole channels and REBUILD a smaller network.
Run:  python q2_prune_structured.py --ratio 0.5 --ft-epochs 10
"""
import argparse

import torch

from common import (build_model, count_macs, count_params, evaluate, get_blocks, get_device,
                    get_test_loader, get_train_loader, model_path, set_seed, shrink_block,
                    train_one_epoch)


def prune_block_channels(block, ratio):
    """Remove `ratio` of the middle channels of one residual block. Returns how many were kept."""
    old_conv1, old_bn1, old_conv2 = block.conv1, block.bn1, block.conv2
    n_out = old_conv1.out_channels
    keep = max(1, int(round(n_out * (1 - ratio))))

    # importance of each channel = sum of absolute weights of its filter (L1-norm)
    importance = old_conv1.weight.data.abs().sum(dim=(1, 2, 3))
    keep_idx = torch.argsort(importance, descending=True)[:keep].sort().values

    shrink_block(block, keep)   # new, thinner layers (random numbers for now)
    # copy the survivors in
    block.conv1.weight.data = old_conv1.weight.data[keep_idx].clone()
    block.bn1.weight.data = old_bn1.weight.data[keep_idx].clone()
    block.bn1.bias.data = old_bn1.bias.data[keep_idx].clone()
    block.bn1.running_mean = old_bn1.running_mean[keep_idx].clone()
    block.bn1.running_var = old_bn1.running_var[keep_idx].clone()
    block.conv2.weight.data = old_conv2.weight.data[:, keep_idx].clone()
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=0.5, help="fraction of middle channels to delete in each block")
    parser.add_argument("--ft-epochs", type=int, default=4, help="fine-tune epochs after pruning")
    parser.add_argument("--lr", type=float, default=0.02)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    train_loader = get_train_loader()
    test_loader = get_test_loader()

    model = build_model()
    model.load_state_dict(torch.load(model_path("fp32"), map_location="cpu"))
    orig_total, _ = count_params(model)
    orig_macs = count_macs(model)
    print(f"Original: {orig_total:,} weights, {orig_macs / 1e6:.1f} M multiply-adds, "
          f"accuracy {evaluate(model, test_loader):.2f}%")

    # prune every block
    keep_counts = [prune_block_channels(b, args.ratio) for b in get_blocks(model)]
    print(f"Channels kept in the middle of each block: {keep_counts}")
    new_total, _ = count_params(model)
    new_macs = count_macs(model)
    print(f"Pruned:   {new_total:,} weights, {new_macs / 1e6:.1f} M multiply-adds "
          f"({100 * (1 - new_macs / orig_macs):.0f}% fewer)")
    print(f"Accuracy right after pruning (before fine-tuning): {evaluate(model, test_loader):.2f}%")

    # fine-tune
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ft_epochs)
    for epoch in range(1, args.ft_epochs + 1):
        loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        print(f"Fine-tune epoch {epoch}/{args.ft_epochs}  loss {loss:.3f}  "
              f"test acc {evaluate(model, test_loader, device):.2f}%")

    # save: the list of channel counts + the weights (so the thin model can be rebuilt)
    torch.save({"keep_counts": keep_counts, "state_dict": model.cpu().state_dict()}, model_path("structured"))
    print(f"\nFINAL accuracy {evaluate(model.cpu(), test_loader):.2f}%   Saved -> {model_path('structured')}")


if __name__ == "__main__":
    main()

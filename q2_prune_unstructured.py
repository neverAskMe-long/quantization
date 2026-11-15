"""
q2_prune_unstructured.py - Unstructured (magnitude-based) pruning.
The weights closest to zero matter least, so we set them to zero.
Run:  python q2_prune_unstructured.py --rounds 4 --amount 0.35 --ft-epochs 2 --final-epochs 5
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from common import (build_model, count_params, evaluate, get_device, get_test_loader,
                    get_train_loader, model_path, set_seed, train_one_epoch)


def sparsity_percent(model):
    total, nonzero = count_params(model)
    return 100.0 * (1 - nonzero / total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3, help="how many pruning rounds")
    parser.add_argument("--amount", type=float, default=0.45,
                        help="fraction of the REMAINING weights removed in each round")
    parser.add_argument("--ft-epochs", type=int, default=1, help="fine-tune epochs after each round")
    parser.add_argument("--final-epochs", type=int, default=2, help="extra fine-tune epochs at the end")
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    train_loader = get_train_loader()
    test_loader = get_test_loader()

    model = build_model()
    model.load_state_dict(torch.load(model_path("fp32"), map_location="cpu"))
    model.to(device)
    print(f"Original accuracy: {evaluate(model, test_loader, device):.2f}%")

    # which weights can be pruned: every Conv2d and Linear layer
    params_to_prune = [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]

    for r in range(1, args.rounds + 1):
        # a. prune: remove the smallest weights (compared across the whole network)
        prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=args.amount)
        acc_after_prune = evaluate(model, test_loader, device)
        print(f"Round {r}/{args.rounds}: sparsity {sparsity_percent(model):.1f}%  "
              f"accuracy right after pruning {acc_after_prune:.2f}%")

        # b. fine-tune (masked weights stay zero)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        for _ in range(args.ft_epochs):
            train_one_epoch(model, train_loader, optimizer, device)
        print(f"          accuracy after fine-tuning {evaluate(model, test_loader, device):.2f}%")

    # extra fine-tuning at the end with a smaller learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr / 5, momentum=0.9, weight_decay=1e-4)
    for _ in range(args.final_epochs):
        train_one_epoch(model, train_loader, optimizer, device)

    # make it permanent: delete the mask and keep the zeros inside the weights
    for module, name in params_to_prune:
        prune.remove(module, name)

    final_acc = evaluate(model, test_loader, device)
    total, nonzero = count_params(model)
    print(f"\nFINAL: accuracy {final_acc:.2f}%  sparsity {sparsity_percent(model):.1f}%  "
          f"({nonzero:,} of {total:,} weights are non-zero)")

    torch.save(model.cpu().state_dict(), model_path("unstructured"))
    print(f"Saved -> {model_path('unstructured')}")


if __name__ == "__main__":
    main()

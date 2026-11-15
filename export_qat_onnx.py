"""
export_qat_onnx.py - Export the QAT model to ONNX by replacing PyTorch's fused fake-quant
modules (not exportable) with standard FakeQuantize modules (exportable).
Run:  python export_qat_onnx.py
"""
import os

import torch
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat
from torch.ao.quantization.fake_quantize import FakeQuantize, FusedMovingAvgObsFakeQuantize

from common import MODELS_DIR, ONNX_DIR, build_model, get_quant_backend
from export_onnx import export_float_model


def unfuse(module):
    """Replace every FusedMovingAvgObsFakeQuantize with a plain FakeQuantize (same numbers)."""
    for name, child in module.named_children():
        if isinstance(child, FusedMovingAvgObsFakeQuantize):
            kwargs = dict(observer=type(child.activation_post_process),
                          quant_min=child.quant_min, quant_max=child.quant_max,
                          dtype=child.dtype, qscheme=child.qscheme)
            if child.qscheme in (torch.per_channel_affine, torch.per_channel_symmetric):
                kwargs["ch_axis"] = child.ch_axis
            new = FakeQuantize(**kwargs)
            new.scale = child.scale.detach().clone()
            new.zero_point = child.zero_point.detach().clone()
            new.disable_observer()      # ranges are final, do not update them
            new.enable_fake_quant()
            setattr(module, name, new)
        else:
            unfuse(child)


def main():
    sd_path = os.path.join(MODELS_DIR, "resnet18_qat_fakequant_state_dict.pth")
    model = build_model()
    model.train()
    model.fuse_model(is_qat=True)
    model.qconfig = get_default_qat_qconfig(get_quant_backend())
    prepare_qat(model, inplace=True)
    model.load_state_dict(torch.load(sd_path, map_location="cpu"))
    unfuse(model)
    model.eval()
    out = os.path.join(ONNX_DIR, "resnet18_qat_int8.onnx")
    export_float_model(model, out)
    print("QAT ONNX exported ->", out)


if __name__ == "__main__":
    main()

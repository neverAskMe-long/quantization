"""
export_onnx.py - Turn the models into ONNX files (run with ONNX Runtime).
Files created in onnx_models/:
  resnet18_fp32.onnx, resnet18_unstructured.onnx, resnet18_structured.onnx,
  resnet18_int8_ort.onnx (INT8 via ONNX Runtime static quantization),
  resnet18_qat_int8.onnx (QAT model with fake-quant nodes, if export works)
If one export fails, the script prints why and continues.
Run:  python export_onnx.py
"""
import os

import numpy as np
import torch
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat

from common import (MODELS_DIR, ONNX_DIR, build_model, get_calibration_loader, get_quant_backend,
                    load_model, set_seed)


def export_float_model(model, out_path):
    model.eval()
    dummy = torch.randn(1, 3, 32, 32)
    kwargs = dict(input_names=["image"], output_names=["logits"],
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(model, dummy, out_path, dynamo=False, **kwargs)   # classic exporter
    except TypeError:   # older PyTorch has no 'dynamo' option
        torch.onnx.export(model, dummy, out_path, **kwargs)
    print(f"  saved {out_path}")


class CalibrationReader:
    """Feeds calibration pictures to ONNX Runtime, one batch at a time."""
    def __init__(self, loader):
        self.batches = iter([x.numpy() for x, _ in loader])

    def get_next(self):
        batch = next(self.batches, None)
        return None if batch is None else {"image": batch}


def make_int8_with_onnxruntime(fp32_onnx, out_path):
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    quantize_static(fp32_onnx, out_path, CalibrationReader(get_calibration_loader(1000)),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=True)
    print(f"  saved {out_path}")


def export_qat_fake_quant(out_path):
    """Rebuild the QAT model (with fake-quant modules), load the trained weights, export."""
    sd_path = os.path.join(MODELS_DIR, "resnet18_qat_fakequant_state_dict.pth")
    model = build_model()
    model.train()
    model.fuse_model(is_qat=True)
    model.qconfig = get_default_qat_qconfig(get_quant_backend())
    prepare_qat(model, inplace=True)
    model.load_state_dict(torch.load(sd_path, map_location="cpu"))
    export_float_model(model.eval(), out_path)


def attempt(title, function, *args):
    print(f"\n{title}")
    try:
        function(*args)
    except Exception as err:   # keep going - report the problem clearly
        print(f"  FAILED: {type(err).__name__}: {str(err)[:300]}")


def main():
    set_seed(42)
    fp32_onnx = os.path.join(ONNX_DIR, "resnet18_fp32.onnx")

    attempt("FP32 baseline -> ONNX", lambda: export_float_model(load_model("fp32"), fp32_onnx))
    attempt("Unstructured-pruned -> ONNX",
            lambda: export_float_model(load_model("unstructured"), os.path.join(ONNX_DIR, "resnet18_unstructured.onnx")))
    attempt("Structured-pruned -> ONNX",
            lambda: export_float_model(load_model("structured"), os.path.join(ONNX_DIR, "resnet18_structured.onnx")))
    attempt("INT8 with ONNX Runtime static quantization", make_int8_with_onnxruntime,
            fp32_onnx, os.path.join(ONNX_DIR, "resnet18_int8_ort.onnx"))
    attempt("QAT (fake-quant) model -> ONNX", export_qat_fake_quant,
            os.path.join(ONNX_DIR, "resnet18_qat_int8.onnx"))

    print("\nONNX files now in onnx_models/:", sorted(os.listdir(ONNX_DIR)))


if __name__ == "__main__":
    main()

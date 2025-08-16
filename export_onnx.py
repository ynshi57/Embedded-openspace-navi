import argparse
import os
import torch
from model_mobilenet_unet import MobileNetV2_UNet


def parse_args():
    parser = argparse.ArgumentParser(description="Export MobileNetV2-UNet to ONNX")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pth state_dict file")
    parser.add_argument("--out", type=str, default=None, help="Output ONNX file path")
    parser.add_argument("--height", type=int, default=256, help="Input height")
    parser.add_argument("--width", type=int, default=256, help="Input width")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version (>=11)")
    parser.add_argument("--no-dynamic", action="store_true", help="Disable dynamic axes for H/W")
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    onnx_out = args.out
    if onnx_out is None:
        base, _ = os.path.splitext(ckpt_path)
        onnx_out = base + ".onnx"
    os.makedirs(os.path.dirname(onnx_out) or ".", exist_ok=True)

    device = torch.device("cpu")

    # Build model and load weights
    model = MobileNetV2_UNet().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Dummy input
    dummy = torch.randn(1, 3, args.height, args.width, device=device)

    # Dynamic axes
    dynamic_axes = None
    if not args.no_dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }

    # Export
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            onnx_out,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

    print(f"✅ Exported ONNX: {onnx_out}")


if __name__ == "__main__":
    main() 
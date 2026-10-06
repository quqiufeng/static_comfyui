"""Export the exact HED / M-LSD ControlNet annotators to ONNX.

One-time dev tool (needs `controlnet-aux`), not a runtime dependency:

    /data/venv/bin/python -m venv --system-site-packages /data/venv-aux
    /data/venv-aux/bin/pip install controlnet-aux
    /data/venv-aux/bin/python export_annotators.py

Writes /data/models/image/annotators/{hed,mlsd}.onnx (+ .onnx.data).
"""
import os, torch, types
from controlnet_aux import HEDdetector, MLSDdetector

out = "/data/models/image/annotators"
os.makedirs(out, exist_ok=True)

# --- HED (Apache2 edge model) ---
hed = HEDdetector.from_pretrained("lllyasviel/Annotators").netNetwork.eval()
if getattr(hed, "forward", None).__func__ is torch.nn.Module.forward:
    hed.forward = hed.__call__  # module overrides __call__, expose as forward for tracing
dummy = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    ys = hed(dummy)
print("HED outputs:", [tuple(y.shape) for y in ys])
torch.onnx.export(hed, dummy, f"{out}/hed.onnx",
                  input_names=["input"], output_names=[f"out{i}" for i in range(len(ys))],
                  dynamic_axes={"input": {0: "b", 2: "h", 3: "w"}}, opset_version=16)
print("hed -> hed.onnx")

# --- MLSD (M-LSD large) ---
mlsd = MLSDdetector.from_pretrained("lllyasviel/Annotators").model.eval()
d2 = torch.randn(1, 4, 512, 512)
with torch.no_grad():
    o = mlsd(d2)
print("MLSD output:", tuple(o.shape))
torch.onnx.export(mlsd, d2, f"{out}/mlsd.onnx",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "b", 2: "h", 3: "w"}}, opset_version=16)
print("mlsd -> mlsd.onnx")
print("done")

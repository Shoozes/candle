"""Export the pinned HF backend's native uint8 TorchVision v2 resize, not v1."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
from manifest import require_reference_environment


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require_reference_environment()
    import torch
    import torchvision
    from torchvision.transforms.v2 import functional as F
    from torchvision.transforms import InterpolationMode
    cases = []
    for w, h, ow, oh in [(7, 5, 4, 3), (7, 5, 11, 9), (48, 32, 16, 16), (384, 160, 416, 192), (1, 5, 1, 9)]:
        image = torch.tensor([[[((x * 37 + y * 19) % 256), ((x * 17 + y * 71) % 256), ((x * 91 + y * 13) % 256)]
                               for x in range(w)] for y in range(h)], dtype=torch.uint8).permute(2, 0, 1)
        output = F.resize(image, [oh, ow], InterpolationMode.BICUBIC, antialias=True)
        # Large case samples still cover every support/edge, without a large fixture.
        indices = list(range(output.numel())) if ow * oh < 1000 else list(range(0, output.numel(), 113))
        values = output.permute(1, 2, 0).contiguous().flatten().tolist()
        cases.append(dict(width=w, height=h, output_width=ow, output_height=oh,
                          indices=indices, rgb=[values[i] for i in indices]))
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(schema="candle.d1_torchvision_bicubic.v2", torch=torch.__version__,
                       torchvision=torchvision.__version__, backend="torchvision.transforms.v2.functional.resize uint8 CPU", cases=cases), stream, separators=(",", ":"))
        stream.write("\n")


if __name__ == "__main__":
    main()

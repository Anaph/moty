"""Resize an image to the NPU encoder's square input and write binary PPM.

  prep_image.py <image> <out.ppm> [size=512]

Bilinear resize to size x size (the HF LFM2-VL processor's resample=2 for
the single-tile case), RGB, uint8."""
import sys
from PIL import Image
src, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 512
im = Image.open(src).convert("RGB")
if im.size != (n, n):
    im = im.resize((n, n), Image.BILINEAR)
with open(out, "wb") as f:
    f.write(b"P6 %d %d 255\n" % (n, n))
    f.write(im.tobytes())
print(out, im.size)

"""
File: unpack_raw10.py
Description: Convert packed 10-bit sensor RAW dumps into the unpacked little-endian
             uint16 .raw format that Infinite-ISP's `InfiniteISP.load_raw()` expects.

`infinite_isp.py` reads a >8-bit .raw with
    np.fromfile(path, dtype=np.uint16).reshape((height, width))
so the file on disk must contain exactly height*width uint16 samples, one per pixel,
with no line padding. Sensor/CSI dumps are usually packed at 10 bits per pixel
(1.25 bytes/px) and may carry a line stride larger than the active width, so they
have to be converted first.

Two packings are supported:

  bitstream (default)
      A plain little-endian 10-bit-per-pixel bit stream. 5 bytes hold 4 pixels:
          p0 =  b0        | (b1 & 0x03) << 8
          p1 = (b1 >> 2)  | (b2 & 0x0F) << 6
          p2 = (b2 >> 4)  | (b3 & 0x3F) << 4
          p3 = (b3 >> 6)  |  b4         << 2
      This is what the D-Robotics / Horizon VIN dumps (pack_mode = 1) contain.

  mipi
      The CSI-2 RAW10 "4 MSB bytes + 1 LSB byte" grouping:
          p_i = b_i << 2 | (b4 >> (2*i)) & 0x03

Usage
    python tools/unpack_raw10.py INPUT.raw OUTPUT.raw --width 1920 --height 1080 \
        [--stride 2400] [--layout bitstream|mipi] [--probe]

`--probe` skips writing and prints black level, saturation and a Bayer-phase
analysis instead, which is the quickest way to check that a layout guess is right:
for a correct unpacking one of the two diagonal pixel pairs (the Gr/Gb pair) has a
mean ratio very close to 1.0 and a spatial CV below ~0.05.
"""

import argparse
import sys

import numpy as np


def unpack(data, width, height, stride, layout):
    """Unpack a packed 10-bit frame into a (height, width) uint16 array."""
    packed_row_bytes = width * 5 // 4
    if width % 4:
        raise ValueError(f"width must be a multiple of 4 for 10-bit packing, got {width}")
    if stride < packed_row_bytes:
        raise ValueError(f"stride {stride} is smaller than the packed row size {packed_row_bytes}")

    expected = stride * height
    if data.size < expected:
        raise ValueError(f"file holds {data.size} bytes, need {expected} for {width}x{height} stride {stride}")
    if data.size > expected:
        print(f"note: ignoring {data.size - expected} trailing bytes", file=sys.stderr)

    rows = data[:expected].reshape(height, stride)[:, :packed_row_bytes]
    grp = rows.reshape(height, -1, 5).astype(np.uint16)
    out = np.empty((height, width), np.uint16)

    if layout == "bitstream":
        out[:, 0::4] = grp[:, :, 0] | ((grp[:, :, 1] & 0x03) << 8)
        out[:, 1::4] = (grp[:, :, 1] >> 2) | ((grp[:, :, 2] & 0x0F) << 6)
        out[:, 2::4] = (grp[:, :, 2] >> 4) | ((grp[:, :, 3] & 0x3F) << 4)
        out[:, 3::4] = (grp[:, :, 3] >> 6) | (grp[:, :, 4] << 2)
    elif layout == "mipi":
        for i in range(4):
            out[:, i::4] = (grp[:, :, i] << 2) | ((grp[:, :, 4] >> (2 * i)) & 0x03)
    else:
        raise ValueError(f"unknown layout {layout!r}")

    return out


def probe(img):
    """Print black level, saturation and a Bayer-phase analysis of an unpacked frame."""
    f = img.astype(np.float32)
    phases = {k: f[k[0]::2, k[1]::2] for k in [(0, 0), (0, 1), (1, 0), (1, 1)]}

    print(f"min={img.min()} max={img.max()} mean={f.mean():.1f}")
    print(f"black level (0.05 pct) = {np.percentile(f, 0.05):.0f}   "
          f"saturated fraction = {(img >= 1023).mean():.4f}")
    print("phase means: " + "  ".join(f"{k}={v.mean():7.1f}" for k, v in phases.items()))

    def blk(a, n=8):
        h, w = a.shape[0] // n * n, a.shape[1] // n * n
        return a[:h, :w].reshape(h // n, n, w // n, n).mean((1, 3))

    b = {k: blk(v) for k, v in phases.items()}
    mask = np.ones_like(b[(0, 0)], bool)
    for v in b.values():
        mask &= (v > 30) & (v < 1000)

    print("\ndiagonal pair test - the Gr/Gb pair has ratio ~1.0, CV < 0.05, corr > 0.99:")
    for a, c in [((0, 0), (1, 1)), ((0, 1), (1, 0))]:
        r = b[a][mask] / b[c][mask]
        corr = np.corrcoef(b[a][mask], b[c][mask])[0, 1]
        green = "  <- green pair" if r.std() / r.mean() < 0.05 and corr > 0.99 else ""
        print(f"  {a}/{c}: ratio={r.mean():.3f} CV={r.std() / r.mean():.4f} corr={corr:.4f}{green}")
    print("\ngreen on (0,1)/(1,0) -> bayer_pattern is rggb or bggr")
    print("green on (0,0)/(1,1) -> bayer_pattern is grbg or gbrg")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output", nargs="?", help="omit together with --probe")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--stride", type=int, default=None,
                    help="bytes per line in the input file (default: width * 5 // 4)")
    ap.add_argument("--layout", choices=["bitstream", "mipi"], default="bitstream")
    ap.add_argument("--probe", action="store_true", help="analyse instead of writing")
    args = ap.parse_args()

    stride = args.stride if args.stride is not None else args.width * 5 // 4
    data = np.fromfile(args.input, dtype=np.uint8)
    img = unpack(data, args.width, args.height, stride, args.layout)

    if args.probe:
        probe(img)
        return
    if not args.output:
        ap.error("an output path is required unless --probe is given")

    img.astype("<u2").tofile(args.output)
    print(f"wrote {args.output}: {args.width}x{args.height} uint16, {img.nbytes} bytes")


if __name__ == "__main__":
    main()

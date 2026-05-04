#!/usr/bin/env python3
"""
Boxcar (moving-average) smoothing for 2D seismic data.

Algorithm:
    Boxcar filter slides a window of length (1+2k) along each axis,
    replacing each sample with the average of its neighbors.
    Edge samples are handled by repeating boundary values ("repeat" policy).

    Triangle smoothing = two boxcar passes (window 1+2k → effective triangle 1+4k).

Usage:
    python boxcar_smoothing.py -i input.bin -o output.bin -n1 501 -n2 6801 -r1 0 -r2 79 -nrep 7
    python boxcar_smoothing.py -i input.bin  # uses defaults: r1=0, r2=79, nrep=7, output=smoothed.bin
"""

import argparse
import numpy as np


def boxcar_smoothing(data: np.ndarray, rect: list, nrep: int = 1) -> np.ndarray:
    """Apply multi-dimensional boxcar smoothing."""
    result = data.copy().astype(np.float32)
    for _ in range(nrep):
        for axis in range(data.ndim):
            if rect[axis] > 0:
                result = _boxcar_axis(result, axis, rect[axis])
    return result


def _boxcar_axis(data: np.ndarray, axis: int, radius: int) -> np.ndarray:
    """1D boxcar (moving average) along one axis using cumulative sum."""
    nx = data.shape[axis]
    L = 1 + 2 * radius
    smoothed = np.zeros_like(data, dtype=np.float32)

    it_shape = data.shape[:axis] + data.shape[axis + 1:]
    it = np.nditer(np.zeros(it_shape, dtype=np.uint8), flags=["multi_index"])

    while not it.finished:
        idx = list(it.multi_index)
        idx.insert(axis, slice(None))

        x = data[tuple(idx)].astype(np.float32)
        x_padded = np.pad(x, (radius, radius), mode="edge")

        csum = np.insert(np.cumsum(x_padded, dtype=np.float64), 0, 0.0)
        smoothed[tuple(idx)] = ((csum[L:L + nx] - csum[0:nx]) / L).astype(np.float32)

        it.iternext()

    return smoothed


def main():
    parser = argparse.ArgumentParser(description="Boxcar smoothing for 2D seismic binary data")
    parser.add_argument("-i", "--input", required=True, help="Input binary file (float32, Fortran order)")
    parser.add_argument("-o", "--output", default="smoothed.bin", help="Output binary file (default: smoothed.bin)")
    parser.add_argument("-n1", type=int, default=501, help="Axis 1 size / depth samples (default: 501)")
    parser.add_argument("-n2", type=int, default=6801, help="Axis 2 size / x-distance samples (default: 6801)")
    parser.add_argument("-r1", type=int, default=5, help="Smoothing radius axis 1 / depth (default: 0 = none)")
    parser.add_argument("-r2", type=int, default=79, help="Smoothing radius axis 2 / x-distance (default: 79)")
    parser.add_argument("-nrep", type=int, default=7, help="Number of smoothing passes (default: 7)")
    args = parser.parse_args()

    data = np.fromfile(args.input, dtype="float32").reshape(args.n1, args.n2, order="F")

    result = boxcar_smoothing(data, [args.r1, args.r2], nrep=args.nrep)

    result.astype("float32").flatten(order="F").tofile(args.output)
    print(f"Done: {args.input} -> {args.output} | shape=({args.n1},{args.n2}) rect=[{args.r1},{args.r2}] nrep={args.nrep}")


if __name__ == "__main__":
    main()

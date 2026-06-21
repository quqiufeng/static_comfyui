#!/usr/bin/env python3
"""compare_bin.py — 对比两个 float32 bin 文件，输出 max/mean diff"""
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('a', type=str)
    parser.add_argument('b', type=str)
    parser.add_argument('--shape', type=str, default=None, help='e.g. 1,77,2048')
    parser.add_argument('--atol', type=float, default=1e-3)
    args = parser.parse_args()

    a = np.fromfile(args.a, dtype=np.float32)
    b = np.fromfile(args.b, dtype=np.float32)

    if args.shape:
        shape = tuple(int(x) for x in args.shape.split(','))
        a = a.reshape(shape)
        b = b.reshape(shape)

    diff = np.abs(a - b)
    print(f"shape: {a.shape} {b.shape}")
    print(f"a sum: {a.sum():.6f}, b sum: {b.sum():.6f}")
    print(f"max abs diff: {diff.max():.6e}")
    print(f"mean abs diff: {diff.mean():.6e}")
    print(f"rel max diff: {(diff / (np.abs(a) + 1e-8)).max():.6e}")
    print(f"match within atol={args.atol}: {np.allclose(a, b, atol=args.atol)}")

if __name__ == '__main__':
    main()

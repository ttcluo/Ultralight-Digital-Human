#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 aud_wenet.npy 转为 Android 可读的二进制格式。
格式：4 个 int32 (num_frames, dim1, dim2, dim3) + float32 数据
"""
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npy_path', type=str, help='aud_wenet.npy 路径')
    parser.add_argument('--output', type=str, help='输出 .bin 路径，默认 npy_path 替换为 .bin')
    args = parser.parse_args()

    arr = np.load(args.npy_path).astype(np.float32)
    out_path = args.output or args.npy_path.replace('.npy', '.bin')

    with open(out_path, 'wb') as f:
        dims = list(arr.shape) + [1] * (4 - len(arr.shape))
        for d in dims[:4]:
            f.write(np.int32(d).tobytes())
        arr.tofile(f)

    print(f"[OK] {arr.shape} -> {out_path}")


if __name__ == '__main__':
    main()

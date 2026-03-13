#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从完整训练数据中均匀采样 N 张图片和 landmarks，输出重编号的 Android assets。
"""
import argparse
import os
import shutil


def get_args():
    parser = argparse.ArgumentParser(description='导出 Android 数字人 assets 子集')
    parser.add_argument('--dataset', type=str, required=True, help='训练数据目录 (含 full_body_img/, landmarks/)')
    parser.add_argument('--count', type=int, default=50, help='采样数量')
    parser.add_argument('--output', type=str, default='./android_assets', help='输出目录')
    return parser.parse_args()


def main():
    args = get_args()
    img_dir = os.path.join(args.dataset, 'full_body_img')
    lms_dir = os.path.join(args.dataset, 'landmarks')
    out_img_dir = os.path.join(args.output, 'full_body_img')
    out_lms_dir = os.path.join(args.output, 'landmarks')

    if not os.path.exists(img_dir):
        print(f"[ERROR] 目录不存在: {img_dir}")
        return 1
    if not os.path.exists(lms_dir):
        print(f"[ERROR] 目录不存在: {lms_dir}")
        return 1

    img_files = sorted(
        [f for f in os.listdir(img_dir) if f.endswith('.jpg')],
        key=lambda x: int(x.replace('.jpg', '')) if x.replace('.jpg', '').isdigit() else 0
    )
    max_idx = int(img_files[-1].replace('.jpg', '')) if img_files else -1
    total = max_idx + 1

    if total < 2:
        print(f"[ERROR] 图片数量不足: {total}")
        return 1

    count = min(args.count, total)
    step = max(1, (total - 1) / (count - 1)) if count > 1 else 0
    indices = [int(i * step) for i in range(count)]

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lms_dir, exist_ok=True)

    for new_idx, old_idx in enumerate(indices):
        src_img = os.path.join(img_dir, f'{old_idx}.jpg')
        src_lms = os.path.join(lms_dir, f'{old_idx}.lms')
        dst_img = os.path.join(out_img_dir, f'{new_idx}.jpg')
        dst_lms = os.path.join(out_lms_dir, f'{new_idx}.lms')

        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        else:
            print(f"[WARN] 跳过不存在的图片: {src_img}")
        if os.path.exists(src_lms):
            shutil.copy2(src_lms, dst_lms)
        else:
            print(f"[WARN] 跳过不存在的 landmarks: {src_lms}")

    size_mb = sum(os.path.getsize(os.path.join(out_img_dir, f)) for f in os.listdir(out_img_dir)) / (1024 * 1024)
    print(f"[OK] 导出 {count} 张 -> {args.output}/")
    print(f"     体积约 {size_mb:.1f} MB")
    return 0


if __name__ == '__main__':
    exit(main())

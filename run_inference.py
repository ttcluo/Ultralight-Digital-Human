#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数字人推理一键脚本：提取音频特征 -> 视频推理 -> 音视频合成
"""
import argparse
import os
import subprocess
import sys


def get_args():
    parser = argparse.ArgumentParser(description='数字人推理：输入音频生成口型视频')
    parser.add_argument('--audio', type=str, required=True, help='输入音频路径 (.wav, 16kHz)')
    parser.add_argument('--dataset', type=str, required=True, help='训练数据目录 (含 full_body_img/, landmarks/)')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型 checkpoint 路径 (.pth)')
    parser.add_argument('--output', type=str, default='result.mp4', help='输出视频路径')
    parser.add_argument('--asr', type=str, default='wenet', choices=['wenet', 'hubert'])
    return parser.parse_args()


def main():
    args = get_args()
    project_root = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.abspath(args.audio)
    output_path = os.path.abspath(args.output)
    video_only_path = output_path.rsplit('.', 1)[0] + '_video_only.mp4'

    if not os.path.exists(audio_path):
        print(f"[ERROR] 音频文件不存在: {audio_path}")
        sys.exit(1)
    if not audio_path.lower().endswith('.wav'):
        print("[ERROR] 音频需为 .wav 格式，采样率 16kHz")
        sys.exit(1)

    npy_path = audio_path.replace('.wav', '_wenet.npy') if args.asr == 'wenet' else audio_path.replace('.wav', '_hu.npy')

    print("[1/3] 提取音频特征...")
    ret = subprocess.run(
        [sys.executable, os.path.join(project_root, 'data_utils', 'wenet_infer.py'), audio_path]
        if args.asr == 'wenet'
        else [sys.executable, os.path.join(project_root, 'data_utils', 'hubert.py'), '--wav', audio_path],
        cwd=project_root,
    )
    if ret.returncode != 0:
        print("[ERROR] 音频特征提取失败")
        sys.exit(1)
    if not os.path.exists(npy_path):
        print(f"[ERROR] 特征文件未生成: {npy_path}")
        sys.exit(1)

    print("[2/3] 视频推理...")
    ret = subprocess.run(
        [
            sys.executable, os.path.join(project_root, 'inference.py'),
            '--asr', args.asr,
            '--dataset', os.path.abspath(args.dataset),
            '--audio_feat', npy_path,
            '--save_path', video_only_path,
            '--checkpoint', os.path.abspath(args.checkpoint),
        ],
        cwd=project_root,
    )
    if ret.returncode != 0:
        print("[ERROR] 视频推理失败")
        sys.exit(1)

    print("[3/3] 音视频合成...")
    ret = subprocess.run(
        ['ffmpeg', '-y', '-i', video_only_path, '-i', audio_path, '-c:v', 'libx264', '-c:a', 'aac', output_path],
        capture_output=True,
    )
    if ret.returncode != 0:
        print(f"[ERROR] ffmpeg 合成失败: {ret.stderr.decode()}")
        sys.exit(1)

    try:
        os.remove(video_only_path)
    except OSError:
        pass

    print(f"[OK] 输出: {output_path}")


if __name__ == '__main__':
    main()

"""
使用 ONNX 模型进行批量推理
"""
import argparse
import os
import cv2
import numpy as np
import onnxruntime as ort


def get_args():
    parser = argparse.ArgumentParser(description='ONNX 推理')
    parser.add_argument('--asr', type=str, default='wenet', choices=['wenet', 'hubert'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--audio_feat', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--onnx', type=str, required=True, help='unet.onnx 路径')
    return parser.parse_args()


def get_audio_features(features, index):
    left = index - 4
    right = index + 4
    pad_left = 0
    pad_right = 0
    if left < 0:
        pad_left = -left
        left = 0
    if right > features.shape[0]:
        pad_right = right - features.shape[0]
        right = features.shape[0]
    auds = features[left:right].copy()
    if pad_left > 0:
        auds = np.concatenate([np.zeros_like(auds[:pad_left]), auds], axis=0)
    if pad_right > 0:
        auds = np.concatenate([auds, np.zeros_like(auds[:pad_right])], axis=0)
    return auds


def main():
    args = get_args()
    mode = args.asr

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(args.onnx, providers=providers)
    except Exception:
        session = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])

    input_names = [inp.name for inp in session.get_inputs()]

    audio_feats = np.load(args.audio_feat).astype(np.float32)
    img_dir = os.path.join(args.dataset, 'full_body_img')
    lms_dir = os.path.join(args.dataset, 'landmarks')
    len_img = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')]) - 1
    exm_img = cv2.imread(os.path.join(img_dir, '0.jpg'))
    h, w = exm_img.shape[:2]

    fps = 20 if mode == 'wenet' else 25
    video_writer = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (w, h))

    step_stride = 0
    img_idx = 0

    for i in range(audio_feats.shape[0]):
        if img_idx > len_img - 1:
            step_stride = -1
        if img_idx < 1:
            step_stride = 1
        img_idx += step_stride

        img_path = os.path.join(img_dir, f'{img_idx}.jpg')
        lms_path = os.path.join(lms_dir, f'{img_idx}.lms')

        img = cv2.imread(img_path)
        with open(lms_path, 'r') as f:
            lms_list = [np.array(line.split(), dtype=np.float32) for line in f.read().splitlines()]
        lms = np.array(lms_list, dtype=np.int32)

        xmin, ymin = lms[1][0], lms[52][1]
        xmax = lms[31][0]
        width = xmax - xmin
        ymax = ymin + width

        crop_img = img[ymin:ymax, xmin:xmax]
        h, w = crop_img.shape[:2]
        crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
        crop_img_ori = crop_img.copy()
        img_real_ex = crop_img[4:164, 4:164].copy()
        img_masked = cv2.rectangle(img_real_ex.copy(), (5, 5, 150, 145), (0, 0, 0), -1)

        img_masked = img_masked.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_real_ex = img_real_ex.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_concat = np.concatenate([img_real_ex, img_masked], axis=0)[np.newaxis]

        audio_feat = get_audio_features(audio_feats, i)
        if mode == 'hubert':
            audio_feat = audio_feat.reshape(1, 16, 32, 32)
        else:
            audio_feat = audio_feat.reshape(1, 128, 16, 32)

        ort_inputs = {input_names[0]: img_concat.astype(np.float32), input_names[1]: audio_feat.astype(np.float32)}
        pred = session.run(None, ort_inputs)[0][0]

        pred = (pred.transpose(1, 2, 0) * 255).astype(np.uint8)
        crop_img_ori[4:164, 4:164] = pred
        crop_img_ori = cv2.resize(crop_img_ori, (w, h))
        img[ymin:ymax, xmin:xmax] = crop_img_ori
        video_writer.write(img)

    video_writer.release()


if __name__ == '__main__':
    main()

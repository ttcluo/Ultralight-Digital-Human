from unet import Model
import argparse
import onnx
import torch
import onnxruntime
import numpy as np
import time


def get_args():
    parser = argparse.ArgumentParser(description='Export PyTorch checkpoint to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--output', type=str, default='./unet.onnx', help='Output ONNX path')
    parser.add_argument('--asr', type=str, default='wenet', choices=['wenet', 'hubert'], help='ASR mode, must match training')
    parser.add_argument('--mobile', action='store_true', help='Use mobile channel config [16,32,64,128,256]')
    return parser.parse_args()


def check_onnx(onnx_path, torch_out, torch_in, audio, use_cuda=True):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    try:
        ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    except Exception:
        ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    print(ort_session.get_providers())
    ort_inputs = {
        ort_session.get_inputs()[0].name: torch_in.cpu().numpy(),
        ort_session.get_inputs()[1].name: audio.cpu().numpy()
    }
    t1 = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    t2 = time.time()
    print("onnx time cost:", t2 - t1)
    np.testing.assert_allclose(torch_out[0].cpu().numpy(), ort_outs[0][0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = Model(6, mode=args.asr, mobile=args.mobile).eval().to(device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))

    if args.asr == 'wenet':
        img = torch.zeros([1, 6, 160, 160]).to(device)
        audio = torch.zeros([1, 128, 16, 32]).to(device)
    else:
        img = torch.zeros([1, 6, 160, 160]).to(device)
        audio = torch.zeros([1, 16, 32, 32]).to(device)

    with torch.no_grad():
        torch_out = net(img, audio)
        print("Output shape:", torch_out.shape)
        torch.onnx.export(
            net, (img, audio), args.output,
            input_names=['input', 'audio'],
            output_names=['output'],
            opset_version=11,
            export_params=True
        )

    check_onnx(args.output, torch_out, img, audio, use_cuda=(device == 'cuda'))
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()

# Ultralight Digital Human

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.10-aff.svg"></a>
    <a href="https://github.com/anliyuan/Ultralight-Digital-Human/stargazers"><img src="https://img.shields.io/github/stars/anliyuan/Ultralight-Digital-Human?color=ccf"></a>
  <br>
    <br>
</p>

A Ultralight Digital Human model can run on mobile devices in real time!!!

一个能在移动设备上实时运行的数字人模型,据我所知，这应该是第一个开源的如此轻量级的数字人模型。

Lets see the demo.⬇️⬇️⬇️

先来看个demo⬇️⬇️⬇️

![DigitalHuman](https://github.com/user-attachments/assets/9d0b37ee-2076-4b4f-93ba-eb939a9fb427)

## 如果你视频中声音质量比较差的话，效果大概率不会好。声音质量比较差指的是：1）存在难以忽略的噪声。2）在空旷的房间里录制的视频有回音。3）视频人声不清楚。建议录制视频时候使用外接麦克风，不用拍摄设备自带的麦克风。我自己尝试了声音清晰的情况，不论是wenet还是hubert，效果都非常棒。

## 关于流式推理：

使用流式推理时，建议把静音的图片和对应的关键点放在单独的目录里，img_inference和lms_inference里。

### ！！！！！！建议大家拍摄训练视频的时候前面20秒不说话，但可以做一些小幅度的动作（模拟数字人说话时的动作），这20秒就可以作为流式推理时的素材。！！！！！！

我在代码里加了一些注释，方便大家二次开发

因为一般用到流式推理的场景一般对实时性要求比较高，所以这里我只写了wenet作为音频编码器的情况（实测在2080这样的机器上多个并发时每帧音频处理+视频处理耗时10ms以内，需要将模型转为onnx）。并且根据每个人的使用场景不同，重构代码是必须的，所以我没有做太多的代码优化，这里只提供一些思路给大家参考，如果需要用到hubert作为音频编码器，可以参考其他github的项目。至于C++的推理方法。我大致试了一下，当前方法在ios近两年的设备上实时跑是没什么问题的，大家可以根据dihuman_run.py里的逻辑做翻译，我这里现在有一种能让这个模型跑在更多设备上的方法（效率更高，略微牺牲效果），有人在商用，暂时不做开源。如果大家在使用过程中发现什么问题，请提issue，我会尽力维护这个项目。

## Train

It's so easy to train your own digital human.I will show you step by step.

训练一个你自己的数字人非常简单，我将一步步向你展示。

### install pytorch and other libs

``` bash
conda create -n dh python=3.10
conda activate dh
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install mkl=2024.0
pip install opencv-python
pip install transformers
pip install numpy==1.23.5
pip install soundfile
pip install librosa
pip install onnxruntime
```

I only ran on pytorch==1.13.1, Other versions should also work.

我是在1.13.1版本的pytorch跑的，其他版本的pytorch应该也可以。

Download wenet encoder.onnx from https://drive.google.com/file/d/1e4Z9zS053JEWl6Mj3W9Lbc9GDtzHIg6b/view?usp=drive_link 

and put it in data_utils/

### Data preprocessing

Prepare your video, 3~5min is good. Make sure that every frame of the video has the person's full face exposed and the sound is clear without any noise, put it in a new folder.I will provide a demo video.

准备好你的视频，3到5分钟的就可以，必须保证视频中每一帧都有整张脸露出来的人物，声音清晰没有杂音，把它放到一个新的文件夹里面。我会提供一个demo视频，来自康辉老师的口播，侵删。

First of all, we need to extract audio feature.I'm using 2 different extractor from wenet and hubert, thank them for their great work.

wenet的代码和与训练模型来自:https://github.com/Tzenthin/wenet_mnn

首先我们需要提取音频特征，我用了两个不同的特征提取起，分别是wenet和hubert，感谢他们。

When you using wenet, you neet to ensure that your video frame rate is 20, and for hubert,your video frame rate should be 25.

如果你选择使用wenet的话，你必须保证你视频的帧率是20fps，如果选择hubert，视频帧率必须是25fps。

In my experiments, hubert performs better, but wenet is faster and can run in real time on mobile devices.

在我的实验中，hubert的效果更好，但是wenet速度更快，可以在移动端上实时运行

And other steps are in data_utils/process.py, you just run it like this.

其他步骤都写在data_utils/process.py里面了，没什么特别要注意的。

``` bash
cd data_utils
python process.py YOUR_VIDEO_PATH --asr hubert
```

Then you wait.

然后等它运行完就行了

### train

After the preprocessing step, you can start training the model.

上面步骤结束后，就可以开始训练模型了。

Train a syncnet first for better results.

先训练一个syncnet，效果会更好。

``` bash
cd ..
python syncnet.py --save_dir ./syncnet_ckpt/ --dataset_dir ./data_dir/ --asr hubert
```

Then find a best one（low loss） to train digital human model.

然后找一个loss最低的checkpoint来训练数字人模型。

2025.6.4更新
关于syncnet，看到很多issue里面大家提syncnet写的不对。因为这个项目也没有很明确的指标，在生产中，加不加syncnet对结果影响并不大，视觉上不会看出来什么差异的（在我的大量实验中是这样的）。或者说有没有同学可以提供一个更好的syncnet方法？欢迎PR。

``` bash
cd ..
python train.py --dataset_dir ./data_dir/ --save_dir ./checkpoint/ --asr hubert --use_syncnet --syncnet_checkpoint syncnet_ckpt
```

## inference

### 一键推理（推荐）

**PyTorch 推理**（需 PyTorch 环境）：
```bash
python run_inference.py --audio ./data/preview.wav --dataset ./data/raw --checkpoint checkpoint/195.pth --output result.mp4
```

**ONNX 推理**（无需 PyTorch，适合部署）：
```bash
# 1. 导出 ONNX
python pth2onnx.py --checkpoint checkpoint/195.pth --output unet.onnx --asr wenet

# 2. 推理
python run_inference_onnx.py --audio ./data/preview.wav --dataset ./data/raw --onnx unet.onnx --output result.mp4
```

### 分步推理

提取测试音频特征（音频采样率需 16000）：
```bash
python data_utils/hubert.py --wav your_test_audio.wav   # hubert
python data_utils/wenet_infer.py your_test_audio.wav    # wenet
```

得到 your_test_audio_hu.npy 或 your_test_audio_wenet.npy 后：
```bash
python inference.py --asr wenet --dataset ./your_data_dir/ --audio_feat your_test_audio_wenet.npy --save_path xxx.mp4 --checkpoint your_trained_ckpt.pth
```

音视频合成：
```bash
ffmpeg -i xxx.mp4 -i your_audio.wav -c:v libx264 -c:a aac result_test.mp4
```

### 模型导出 (pth2onnx)

```bash
python pth2onnx.py --checkpoint checkpoint/195.pth --output unet.onnx --asr wenet
# 移动端小模型加 --mobile
python pth2onnx.py --checkpoint checkpoint/195.pth --output unet_mobile.onnx --asr wenet --mobile
```

### Android 部署

见 [android/README.md](android/README.md)。需先运行 `export_avatar_assets.py` 和 `export_audio_feat_bin.py` 准备 assets。

## Enjoy🎉🎉🎉

这个模型是支持流式推理的，但是代码还没有完善，之后我会提上来。

关于在移动端上运行也是没问题的，只需要把现在这个模型通道数改小一点，音频特征用wenet就没问题了。相关代码我也会在之后放上来。

if you have some advice, open an issue or PR.

如果你有改进的建议，可以提个issue或者PR。

If you think this repo is useful to you, please give me a star.

如果你觉的这个repo对你有用的话，记得给我点个star

微信群⬇️⬇️⬇️
<table>
  <tr>
    <td><img src="demo/wechat.jpeg" width="180"/></td>
  </tr>
</table>


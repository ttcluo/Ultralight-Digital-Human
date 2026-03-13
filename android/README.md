# Android 数字人推理

## 准备 assets

将以下文件放入 `app/src/main/assets/`：

```
assets/
├── unet.onnx           # pth2onnx 导出
├── audio_feat.bin      # export_audio_feat_bin.py 从 aud_wenet.npy 导出
├── full_body_img/      # export_avatar_assets.py 导出的 0.jpg, 1.jpg, ...
└── landmarks/          # 0.lms, 1.lms, ...
```

### 导出命令

```bash
# 1. 导出 avatar 子集（在项目根目录）
python export_avatar_assets.py --dataset ./data/raw --count 50 --output ./android_assets

# 2. 导出音频特征（需先运行 run_inference_onnx 得到 aud_wenet.npy）
python export_audio_feat_bin.py ./data/preview_wenet.npy --output audio_feat.bin

# 3. 复制到 assets
cp android_assets/full_body_img/* app/src/main/assets/full_body_img/
cp android_assets/landmarks/* app/src/main/assets/landmarks/
cp unet.onnx app/src/main/assets/
cp audio_feat.bin app/src/main/assets/
```

## 构建

```bash
cd android
export JAVA_HOME=/Users/larry/Library/Java/JavaVirtualMachines/jbr-17.0.14/Contents/Home
./gradlew assembleDebug
```

APK 输出：`app/build/outputs/apk/debug/app-debug.apk`

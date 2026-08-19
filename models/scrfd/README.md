# SCRFD

## 人脸检测[SCRFD](https://github.com/deepinsight/insightface/tree/master/model_zoo)

- [流行的人脸检测识别模型集合下载](https://laolaolulu.github.io/posts/face-model-download/)
- [scrfd face detection model in onnx format](https://github.com/cysin/scrfd_onnx/tree/main)
- [SCRFD face detection TensorRT](https://github.com/namdvt/SCRFD_FaceDetection_TensorRT/tree/master)

### 下载 scrfd_2.5g

Download ckpt of desired models from <https://github.com/deepinsight/insightface/blob/master/detection/scrfd/README.md>

## 创建独立 SCRFD 导出环境

```bash
uv venv scrfd_export_venv --python 3.8
source scrfd_export_venv/bin/activate
```

- 安装 PyTorch
为了导出 ONNX，不需要 GPU，使用 CPU 版本能避开 CUDA 依赖问题，安装 PyTorch 1.8.0 CPU 版本：

```bash

python -m ensurepip --upgrade

python -m pip install \ 
    "pip<24.1" \
    "setuptools<60" \
    wheel

python -m pip install \
    torch==1.8.0+cpu \
    torchvision==0.9.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html
```

- 安装与 PyTorch 1.8.0 对应的预编译 MMCV：

```bash
python -m pip install \
    mmcv-full==1.3.3 \
    -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.8.0/index.html
```

- 安装兼容的导出依赖：

```bash
python -m pip install \
    numpy==1.23.5 \
    onnx==1.10.2 \
    onnx-simplifier==0.3.10 \
    protobuf==3.20.3 \
    opencv-python==4.8.1.78 \
    terminaltables \
    matplotlib \
    pycocotools \
    scipy \
    tqdm
```

- 卸载可能误装的 PyPI 版 mmdet：

```bash
python -m pip uninstall mmdet -y
```

- 安装 InsightFace 自带的旧版 mmdet

```bash

git clone https://github.com/deepinsight/insightface

cd insightface/detection/scrfd

python -m pip install -r requirements/build.txt
python -m pip install -v -e . --no-deps
```

- 验证 ABI 和模块

```bash
python - <<'PY'
import torch
import mmcv
import mmdet
import mmdet.core
from mmcv.ops import nms_match

print("torch:", torch.__version__)
print("mmcv:", mmcv.__version__)
print("mmdet:", mmdet.__version__)
print("mmdet path:", mmdet.__file__)
print("mmcv.ops: OK")
PY
```

输出如下：

```bash
torch: 1.8.0+cpu
mmcv: 1.3.3
mmdet: 2.7.0
mmdet path: /home/xuyao/chj/ws/vision_dev/vision_ws/models/scrfd/insightface/detection/scrfd/mmdet/__init__.py
mmcv.ops: OK
```

- 准备一张导出用图片。导出只用于 tracing，任意 JPEG 即可：

```bash
python - <<'PY'
import cv2
import numpy as np

img = np.zeros((640, 640, 3), dtype=np.uint8)
cv2.imwrite("/tmp/scrfd_export_input.jpg", img)
print("/tmp/scrfd_export_input.jpg")
PY
```

- 导出 ONNX

```bash
cd .../insightface

# SCRFD_2.5G
python detection/scrfd/tools/scrfd2onnx.py \
    detection/scrfd/configs/scrfd/scrfd_2.5g.py \
    /home/xuyao/chj/ws/vision_dev/vision_ws/models/scrfd/SCRFD_2.5G/model.pth \
    --input-img /tmp/scrfd_export_input.jpg \
    --output-file /home/xuyao/chj/ws/vision_dev/vision_ws/models/scrfd/scrfd_2.5g_shape640x640.onnx \
    --shape 640 640

# SCRFD_2.5G_KPS
python detection/scrfd/tools/scrfd2onnx.py \
    detection/scrfd/configs/scrfd/scrfd_2.5g_bnkps.py \
    /home/xuyao/chj/ws/vision_dev/vision_ws/models/scrfd/SCRFD_2.5G_KPS/model.pth \
    --input-img /tmp/scrfd_export_input.jpg \
    --output-file /home/xuyao/chj/ws/vision_dev/vision_ws/models/scrfd/scrfd_2.5g_bnkps_shape640x640.onnx \
    --shape 640 640
```

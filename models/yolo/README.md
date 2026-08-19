# 模型下载与预处理

## 人体检测[YOLO26](https://platform.ultralytics.com/ziheng-fan/yolo26)

### `yolo26m.pt` -> `onnx` -> `TensorRT engine`

- 下载 [yolo26m.pt](https://platform.ultralytics.com/ziheng-fan/yolo26/yolo26m)

- [将 yolo26m.pt 转换为 onnx 格式](https://docs.ultralytics.com/integrations/onnx#what-are-the-advantages-of-using-onnx-runtime-for-deploying-yolo26-models)：

```bash
uv venv venv
source venv/bin/activate
uv pip install ultralytics -i https://mirrors.aliyun.com/pypi/simple/
uv pip install onnxruntime-gpu onnx onnxslim -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir
yolo export model=./yolo/yolo26m.pt format=onnx quantize=16
```

- 将 onnx 模型转换为 TensorRT 引擎(使用系统自定义安装的TensorRT)：

```bash
trtexec --onnx=./yolo/yolo26m_fp16.onnx \
        --saveEngine=./yolo/yolo26m_fp16.engine \
        --builderOptimizationLevel=5

# old version(tensorrt==10.9.0)
/usr/src/tensorrt/bin/trtexec --onnx=./yolo/yolo26m_fp16.onnx \
        --saveEngine=./yolo/yolo26m_fp16.engine \
        --fp16 \
        --builderOptimizationLevel=5
```

### `yolo26m.pt` -> `TensorRT engine`

要求系统的 CUDA 版本, TensorRT 版本和 PyTorch 版本兼容, 否则会报错.

```bash
yolo export model=./yolo/yolo26m.pt format=engine quantize=16
```

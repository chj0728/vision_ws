
# [YOLO26](https://platform.ultralytics.com/ziheng-fan/yolo26)

## (推荐)Export `yolo26m.pt` -> `onnx` -> `engine`

- 下载 [yolo26m.pt](https://platform.ultralytics.com/ziheng-fan/yolo26/yolo26m)

- 安装依赖：

```bash
uv venv venv --system-site-packages # 继承系统安装的 tensorrt
source venv/bin/activate
uv pip install ultralytics onnx onnxslim onnxruntime -i https://mirrors.aliyun.com/pypi/simple/
```

- [将 yolo26m.pt 转换为 onnx 格式](https://docs.ultralytics.com/integrations/onnx#what-are-the-advantages-of-using-onnx-runtime-for-deploying-yolo26-models)：

```bash
yolo export model=./yolo26m.pt format=onnx quantize=16

mv ./yolo26m.onnx ./yolo26m_fp16.onnx
```

- 将 onnx 模型转换为 engine(使用系统自定义安装的TensorRT)

```bash
# new version(tensorrt==11.2.1.2)
trtexec --onnx=./yolo26m_fp16.onnx \
        --saveEngine=./yolo26m_fp16.engine \
        --builderOptimizationLevel=5

# old version(tensorrt==10.9.0)
/usr/src/tensorrt/bin/trtexec --onnx=./yolo26m_fp16.onnx \
        --saveEngine=./yolo26m_fp16.engine \
        --fp16 \
        --builderOptimizationLevel=5
```

## Export `yolo26m.pt` -> `engine`

要求系统的 CUDA 版本, TensorRT 版本和 PyTorch 版本兼容, 否则会报错.

```bash
# yolo26m.pt -> yolo26m.onnx -> yolo26m.fp16.onnx -> yolo26m.engine
yolo export model=./yolo26m.pt format=engine quantize=16

mv ./yolo26m.engine ./yolo26m_fp16.engine
```

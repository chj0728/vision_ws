# 人脸识别

## 下载 arcface_w600k_r50_fp16.onnx

```bash
wget https://huggingface.co/leandro-driguez/swap-faces/resolve/main/arcface_w600k_r50_fp16.onnx
```

## 转换为 engine

```bash
# /usr/src/tensorrt/bin/trtexec --onnx=./arcface_w600k_r50_fp16.onnx \
        --saveEngine=./arcface_w600k_r50_fp16.engine \
        --fp16 \
        --builderOptimizationLevel=5

# new version(tensorrt>=11)
trtexec --onnx=./arcface_w600k_r50_fp16.onnx \
        --saveEngine=./arcface_w600k_r50_fp16.engine \
        --builderOptimizationLevel=5
```

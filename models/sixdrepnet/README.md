# 头部姿态识别

## 转换为 engine

```bash
# /usr/src/tensorrt/bin/trtexec --onnx=./SixDRepNet.onnx \
        --saveEngine=./SixDRepNet_fp16.engine \
        --fp16 \
        --builderOptimizationLevel=5
# new version(tensorrt>=11)
trtexec --onnx=./SixDRepNet.onnx \
        --saveEngine=./SixDRepNet.engine \
        --builderOptimizationLevel=5
```

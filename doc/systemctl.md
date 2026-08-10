Optimized tool selection这两个服务可通过 `systemctl` 管理。

### 查看状态

```bash
sudo systemctl status orbbec-camera.service
sudo systemctl status human-face-fusion.service
```

查看是否已设置开机启动：

```bash
systemctl is-enabled orbbec-camera.service
systemctl is-enabled human-face-fusion.service
```

查看运行日志：

```bash
sudo journalctl -u orbbec-camera.service -f
sudo journalctl -u human-face-fusion.service -f
```

查看本次开机以来的日志：

```bash
sudo journalctl -b -u orbbec-camera.service
sudo journalctl -b -u human-face-fusion.service
```

### 停止服务

由于 `human-face-fusion.service` 依赖相机服务，建议先停止推理服务：

```bash
sudo systemctl stop human-face-fusion.service
sudo systemctl stop orbbec-camera.service
```

只停止当前运行，不会取消下次开机启动。

### 关闭开机启动

```bash
sudo systemctl disable human-face-fusion.service
sudo systemctl disable orbbec-camera.service
```

同时立即停止并取消开机启动：

```bash
sudo systemctl disable --now human-face-fusion.service
sudo systemctl disable --now orbbec-camera.service
```

### 重新开启

设置开机启动并立即运行：

```bash
sudo systemctl enable --now orbbec-camera.service
sudo systemctl enable --now human-face-fusion.service
```

### 完全禁止启动

如果希望服务不能被手动或依赖关系启动：

```bash
sudo systemctl mask human-face-fusion.service
sudo systemctl mask orbbec-camera.service
```

恢复：

```bash
sudo systemctl unmask orbbec-camera.service
sudo systemctl unmask human-face-fusion.service
```

修改 `.service` 文件并复制到 system 后，需要执行：

```bash
sudo systemctl daemon-reload
sudo systemctl restart orbbec-camera.service
sudo systemctl restart human-face-fusion.service
```

注意：由于 `human-face-fusion.service` 中配置了：

```ini
Requires=orbbec-camera.service
After=orbbec-camera.service
```

启动人脸融合服务时会自动启动相机服务；停止或禁用相机服务前，最好先停止人脸融合服务。

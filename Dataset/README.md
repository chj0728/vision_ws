# 测试数据集

## [TUM RGB-D Dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download?)

- Sequence 'freiburg3_sitting_static'

```bash
# ROS bag 
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_static.bag
```

- ROS1 bag 转换成 ROS2 bag

```bash
uv venv venv
source venv/bin/activate
uv pip install rosbags -i https://mirrors.aliyun.com/pypi/simple/

rosbags-convert \
--src rgbd_dataset_freiburg3_sitting_static.bag \
--dst rgbd_dataset_freiburg3_sitting_static_ros2
```

# trt_infer

基于 TensorRT、CUDA 和 OpenCV 的视觉推理组件集合，包含人脸分析与目标检测能力。

## 模块说明

| 模块 | 主要类/数据结构 | 功能 |
| --- | --- | --- |
| `scrfd_trt` | `SCRFD_TRT`、`FaceObject` | 使用 SCRFD 检测图像中的人脸，输出人脸框、5 点关键点和置信度；支持 InsightFace 与 Namdvt 两种预处理方式。 |
| `arcface_trt` | `ArcFaceTRT`、`FaceEmbedding` | 对基于关键点对齐后的人脸提取 512 维 L2 归一化特征；提供人脸对齐、特征归一化和余弦相似度计算。 |
| `arcface_trt` | `FaceDatabase`、`PersonRecord`、`FaceMatch` | 使用 SQLite 管理人脸库；加载特征至内存进行快速比对，并支持人员注册、命名、识别和访问记录更新。 |
| `gender_trt` | `GenderAgeTRT`、`GenderResult` | 基于对齐人脸预测性别、置信度和年龄；兼容常见的多种 TensorRT 输出布局。 |
| `sixdrepnet_trt` | `SixDRepNet_TRT`、`HeadPose` | 估计人脸偏航、俯仰和滚转角；支持在图像中绘制头部姿态坐标轴。 |
| `yolo_trt` | `YOLOEngine`、`Detection` | 执行 YOLO 目标检测，输出类别、置信度和边界框；支持 GPU 预处理、NMS 后处理及深度图距离采样。 |

## 人脸分析流程

1. `SCRFD_TRT::detect` 检测人脸，得到 `FaceObject`。
2. `ArcFaceTRT::alignFace` 使用 5 点关键点生成人脸对齐图。
3. `ArcFaceTRT::extractEmbedding` 提取 512 维人脸特征。
4. `FaceDatabase::identify` 按余弦相似度完成人员识别；未识别人员可通过 `registerPerson` 入库。
5. `GenderAgeTRT::predict` 与 `SixDRepNet_TRT::predict` 可复用对齐人脸，分别生成年龄性别和头部姿态结果。

## 目标检测与距离估计

`YOLOEngine::infer` 完成普通目标检测。`YOLOEngine::inferWithDepth` 在同一 GPU 链路中完成图像预处理、推理、后处理和深度图采样，并将距离写入 `Detection::distance`。

## 人脸数据库时间字段（2026-09-11）

实现见 [face_database.h](include/arcface_trt/face_database.h) 和 [face_database.cpp](include/arcface_trt/face_database.cpp)。数据库时间列采用 `TEXT`，按照**运行进程使用的系统时区**保存 `YYYY-MM-DD HH:MM:SS.SSS`。例如系统为 `Asia/Shanghai` 时，直接显示北京时间 `2026-09-11 11:04:05.123`。末尾三位表示毫秒。SQLite 使用系统本地时区规则；进程若设置了 `TZ` 环境变量，则以该变量指定的时区为准。

| 表 | 字段 | 数据库类型 | 含义及更新时机 |
| --- | --- | --- | --- |
| `persons` | `first_seen` | `TEXT NOT NULL` | 首次注册时间，后续访问不修改 |
| `persons` | `last_seen` | `TEXT NOT NULL` | 注册时初始化，`touchPerson()` 时更新 |
| `face_embeddings` | `captured_at` | `TEXT NOT NULL` | 特征入库时间，当前使用注册时刻，不是相机采集时间 |

新建时间列的缺省值为 Unix 起点在系统时区的表示，例如北京时间为 `1970-01-01 08:00:00.000`，对应旧版缺省值 `0`。已有 TEXT 列的 SQL 缺省定义保持原样；应用注册时总是显式写入时间，不依赖该缺省值。一次注册使用同一个时刻填写人物和全部特征的时间；识别算法、特征 BLOB、UUID、姓名及访问次数含义不变。

### 上层接口兼容

`open()`、`identify()`、`registerPerson()`、`updateName()`、`touchPerson()` 的签名和正常调用方式不变。`PersonRecord.first_seen/last_seen` 仍为 `int64_t` Unix 毫秒，读库时将文本转换回来；上层无需改用字符串。`nowMs()` 继续获取毫秒，写入 SQL 时通过 `strftime(..., 'unixepoch', 'localtime')` 格式化为本地时间，读入时通过 `utc` 修饰符转换回 Unix 毫秒。

直接读取 SQLite 的外部脚本需要按新类型处理时间，例如：

```sql
SELECT uuid, name, first_seen, last_seen, visit_count FROM persons;

-- 已经是系统本地时间，无需再加 8 小时。
SELECT person_uuid, captured_at FROM face_embeddings;
```

### 旧库迁移

`open()` 创建缺失表后依次调用 `migrateTimeColumns()` 和 `migrateToLocalTime()`，再加载内存人脸库：

1. 检查实际列类型。三个时间列已经是 `TEXT` 时不重建表。
2. 发现旧 `INTEGER` 时间列时，暂时关闭外键检查，开启写事务；创建新表并将 Unix 毫秒转换为 UTC 文本。部分列已经为 `TEXT` 时保留其文本并检查格式。
3. 保留原有 UUID、姓名、访问次数、特征 ID、特征 BLOB、姿态及置信度；重建关联索引和触发器，并保留自增 ID 的历史最高值。
4. 校验时间格式及外键关系，成功后提交并恢复外键检查。这一步仍沿用旧版 UTC 文本作为中间格式。错误时回滚当前事务，关闭连接，`open()` 返回 `-1`，`isOpen()` 为 false。

5. 检查 `face_db_metadata` 中的 `time_format`。无标记表示旧版 UTC 文本（包括刚转换的整数库），在独立事务中将三个字段转为系统本地时间，并写入 `system_local_ms` 标记；已存在该标记时不再转换，避免每次打开重复偏移。

时间文本不附带时区偏移。部署后应保持数据库使用同一系统时区；修改系统时区或将库迁移到其他时区的设备时，需要单独转换历史时间，重启不会自动重写已标记的数据。采用夏令时的时区在回拨时可能出现重复本地时间，当前文本格式不能区分这一小时内的两个相同时刻。无标记但由其他工具写入的本地时间库不能直接按本项目旧 UTC 库迁移。

迁移针对本项目既有表布局；额外列、异常时间格式、临时表名冲突或无法重建的自定义依赖会使迁移失败，不应直接忽略失败继续识别。旧版程序不理解新的文本时间，因此升级后不应与旧版程序同时读写同一库。正式部署前保留旧库备份；迁移会在应用下次打开数据库时执行，本次代码测试不直接操作业务数据库。

SQLite 没有专用日期时间存储类型，本文采用其支持的日期文本与 `strftime()` 转换；表类型变更使用事务重建方式。参考 [SQLite 日期时间函数](https://www.sqlite.org/lang_datefunc.html) 和 [SQLite 表结构变更说明](https://www.sqlite.org/lang_altertable.html)。

## 辅助文件

- `scrfd_gpu_ops.cu/.h`：SCRFD 的 CUDA 图像预处理和候选框处理实现。
- `yolo_gpu_preprocess.cu/.h`：YOLO 的 CUDA 预处理、候选框生成和深度距离计算实现。
- `logging.h`、`macros.h`：TensorRT 日志和通用辅助定义。

## 引用

- TensorRT 实现编译为共享库 libtrt_infer.so。
- 导出 CMake target、公共头文件和依赖，其他 ROS 2 包可直接引用。
- 自动识别平台：
  - aarch64/arm64：按 Jetson AGX Orin 配置，默认 CUDA Compute Capability 为 87。
  - x86_64/amd64：使用 x86_64 TensorRT 库目录，由 CUDA 编译器选择默认 GPU 架构。
- 自动查找 CUDA、TensorRT、OpenCV、SQLite3 和 UUID。
- 安装时只发布 .h、.hpp 公共头文件，不安装 .cpp、.cu。

其他包可按以下方式引用：

```cmake
find_package(trt_infer REQUIRED)

target_link_libraries(your_target 
trt_infer::trt_infer)
```

## ArcFace Embedding NaN 修复 - 2026-08-21

测试 ArcFace Pipeline 时，发现输出的 512 维人脸特征全部为 `NaN`：

```text
Embedding for track ID 0: [nan, nan, nan, nan, nan, ...]
```

### 原因

- 原实现固定按 FP32 分配和复制 TensorRT 输入、输出缓冲区。
- 当前 ArcFace 模型可能使用 FP16 输入或输出，FP16 数据被按 FP32 读取后会产生错误数值或 `NaN`。
- 原 `l2Normalize()` 未检查 `NaN`、`Inf` 和零范数，一个异常分量会使整个 Embedding 在归一化后变成 `NaN`。
- CUDA 异步推理完成后的同步状态未检查，运行时错误可能被当作推理成功。
- 无效 Embedding 仍会进入消息缓冲，并可能被自动注册到 SQLite 人脸库。

### TensorRT 推理修复

更新 `arcface_trt.h` 和 `arcface_trt.cpp`：

- 读取并验证 TensorRT Engine 的真实输入、输出数据类型。
- 同时支持 FP16 和 FP32 I/O。
- FP16 输入执行 `float -> half` 转换，FP16 输出执行 `half -> float` 转换。
- 验证 Engine 仅包含一个输入和一个输出。
- 验证输入形状为 `[1, 3, 112, 112]`，输出元素数量为 512。
- 检查动态输入形状设置和 Tensor 地址绑定结果。
- 检查 H2D、D2H、`enqueueV3()` 和 `cudaStreamSynchronize()` 的执行结果。
- 推理失败时将输出特征清零并返回 `false`。
- Engine 加载时输出实际 Tensor 名称及 I/O 数据类型，便于确认模型接口。

### Embedding 有效性保护

- `l2Normalize()` 改为返回归一化是否成功。
- 归一化前检查所有特征值是否为有限数。
- 使用双精度累计 L2 范数，拒绝非有限或接近零的范数。
- 归一化失败时清零整个 Embedding，阻止其继续发布、识别或注册。
- `cosineSim()` 遇到无效特征时返回不匹配结果。
- 部分特征值仅在完整校验成功后打印，不再输出或注册 `NaN`。

### 人脸数据库保护

更新 `face_database.cpp`：

- 从 SQLite 加载时跳过 `NaN`、`Inf` 和零范数特征。
- 身份查询前重新验证并归一化查询特征。
- 注册人物前过滤无效 Embedding。
- 没有任何有效特征时拒绝创建人物记录。
- 已经写入数据库的无效特征不会再加载到内存人脸库。

> 修复前自动注册的测试人物可能已包含无效 Embedding，重新测试前应备份并清理对应测试数据库。

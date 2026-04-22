# 附件：代码实现说明

## A.1 附件说明

本附件用于对课程项目中的主要代码文件进行分文件、分阶段说明，重点解释每个脚本在整套暗光目标检测流水线中的作用、主要处理步骤、关键函数、输入输出关系以及实现时特别注意的技术细节。正文负责介绍实验背景、方法和结果，本附件则更偏向“代码层面的实现解释”，便于在答辩或教师检查代码时快速说明“每一份代码具体做了什么”。

本项目的核心思路是构建两条级联流水线：

- 基础方案：`VOC_Dark -> CLAHE -> YOLOv5su`
- 前沿方案：`VOC_Dark -> Zero-DCE++ -> YOLO26n`

围绕这个主线，代码可以按功能拆成四个阶段：

1. 数据下载与标注转换
2. 暗光数据构建
3. 图像增强
4. 训练、验证与预测

---

## A.2 项目代码结构

本项目中最核心的代码文件如下：

- [01_voc_download_and_yolo.py](<C:/CNN/cv_project/scripts/01_voc_download_and_yolo.py:1>)
- [02_degradation.py](<C:/CNN/cv_project/scripts/02_degradation.py:1>)
- [03_clahe_process.py](<C:/CNN/cv_project/scripts/03_clahe_process.py:1>)
- [04_zerodce_process.py](<C:/CNN/cv_project/scripts/04_zerodce_process.py:1>)
- [eval_pipeline.py](<C:/CNN/cv_project/eval_pipeline.py:1>)
- [run_autodl.sh](<C:/CNN/cv_project/run_autodl.sh:1>)
- [test_step1_scripts.py](<C:/CNN/cv_project/tests/test_step1_scripts.py:1>)

辅助配置文件包括：

- [dark.yaml](<C:/CNN/cv_project/config/dark.yaml:1>)
- [clahe.yaml](<C:/CNN/cv_project/config/clahe.yaml:1>)
- [zdcepp.yaml](<C:/CNN/cv_project/config/zdcepp.yaml:1>)

---

## A.3 数据下载与标注转换

### A.3.1 文件：`01_voc_download_and_yolo.py`

文件位置：

- [01_voc_download_and_yolo.py](<C:/CNN/cv_project/scripts/01_voc_download_and_yolo.py:1>)

### A.3.1.1 作用

这个脚本负责完成整个项目的第一步：准备标准 VOC2007 数据，并把原始 VOC XML 标注转换成 YOLO 检测器可直接读取的 TXT 标注格式。它是整个项目的数据入口。

### A.3.1.2 分阶段说明

第一阶段：准备类别映射

- 在文件开头显式定义 VOC20 类别顺序 `CLASSES`
- 同时建立 `CLASS_TO_ID`

这一步非常关键，因为如果类别顺序不固定，后续训练和评估的类别 id 就会错乱，mAP 会直接失真。

第二阶段：准备 VOC 数据源

- 优先检查本地是否已有 `VOCtrainval_06-Nov-2007.zip` 和 `VOCtest_06-Nov-2007.zip`
- 如果有，就直接解压
- 如果没有，再退回到 `torchvision.datasets.VOCDetection` 下载

第三阶段：读取划分列表

- 从 `ImageSets/Main/` 中读取 `trainval.txt` 和 `test.txt`
- 获取每个 split 中的图像 id 列表

第四阶段：VOC XML 转 YOLO TXT

- 解析每张图对应的 XML 标注
- 提取目标框绝对坐标 `[xmin, ymin, xmax, ymax]`
- 调用 `voc_box_to_yolo()` 转成 YOLO 格式

第五阶段：输出新目录结构

- 复制原始图像到 `images/trainval` 和 `images/test`
- 保存标签到 `labels/trainval` 和 `labels/test`
- 复制 XML 到 `annotations/`
- 写 `splits/*.txt`
- 写 `classes.txt`
- 写 `dataset_summary.json`

### A.3.1.3 关键函数

`class_name_to_id()`

- 功能：把类别名映射为固定 id
- 关键点：不允许出现不在 VOC20 中的类别

`voc_box_to_yolo()`

- 功能：将 VOC 绝对框转换为 YOLO 归一化框
- 关键点：
  - 使用图像真实宽高
  - 做归一化
  - 做裁剪，保证结果在 `[0, 1]`

`parse_annotation()`

- 功能：读取 XML，提取图像尺寸和所有目标框
- 关键点：
  - 支持过滤 `difficult`
  - 逐个目标转成 YOLO 标签

`copy_and_convert_split()`

- 功能：按 `trainval` 或 `test` 复制图像、复制 XML、写标签
- 关键点：同时统计每个 split 的类别直方图和空标签情况

### A.3.1.4 实现中的关键点

最关键的风险有两个：

1. 宽高搞反
2. 归一化后坐标越界

这个脚本通过固定函数封装和裁剪逻辑避免了这些问题，是整个项目里最重要的数据安全保障模块之一。

### A.3.1.5 输入输出

输入：

- VOC2007 原始压缩包或 torchvision 下载的数据

输出：

- `datasets/VOC_Original/images/*`
- `datasets/VOC_Original/labels/*`
- `datasets/VOC_Original/annotations/*`
- `datasets/VOC_Original/classes.txt`
- `datasets/VOC_Original/dataset_summary.json`

---

## A.4 暗光数据构建

### A.4.1 文件：`02_degradation.py`

文件位置：

- [02_degradation.py](<C:/CNN/cv_project/scripts/02_degradation.py:1>)

### A.4.1.1 作用

这个脚本负责将正常光照的 `VOC_Original` 离线退化为暗光版本 `VOC_Dark`，为后续两条增强方案提供统一输入。

### A.4.1.2 分阶段说明

第一阶段：读取原始图像

- 遍历 `images/trainval` 和 `images/test`
- 读取每张原始图像

第二阶段：为每张图生成稳定随机参数

- 通过图像路径和全局种子构造哈希
- 为每张图稳定地产生一组 `gamma` 和 `noise_std`

这样设计的好处是：多次重跑时，同一张图会退化成同样的暗光图，保证实验可复现。

第三阶段：执行退化

- 先把图像转成 `float32` 并归一化到 `[0, 1]`
- 执行 Gamma 变换
- 叠加高斯噪声
- 再裁剪并转回 `uint8`

第四阶段：保存退化结果与清单

- 输出到 `VOC_Dark/images/*`
- 复制标签、标注和 split 文件
- 写 `degradation_manifest.json`

### A.4.1.3 关键函数

`apply_low_light_degradation()`

- 功能：执行暗光退化
- 关键点：
  - 必须在 `float32` 域计算
  - 禁止在 `uint8` 上直接做 Gamma 和加噪

`build_per_image_rng()`

- 功能：给每张图生成稳定的随机参数
- 关键点：同一图像跨次运行结果一致

`process_split()`

- 功能：对一个 split 完成批量退化
- 关键点：同时统计均值、上下界等信息

### A.4.1.4 实现中的关键点

这个脚本最重要的是避免数值溢出。因为如果在 `uint8` 上直接做 Gamma 和噪声，图像很容易出现严重的颜色断层和脏噪点，导致后续增强和检测都不可靠。

### A.4.1.5 输入输出

输入：

- `datasets/VOC_Original`

输出：

- `datasets/VOC_Dark`
- `degradation_manifest.json`

---

## A.5 基础增强实现

### A.5.1 文件：`03_clahe_process.py`

文件位置：

- [03_clahe_process.py](<C:/CNN/cv_project/scripts/03_clahe_process.py:1>)

### A.5.1.1 作用

这个脚本负责对 `VOC_Dark` 做传统图像增强，生成 `VOC_CLAHE`，是基础方案的增强模块。

### A.5.1.2 分阶段说明

第一阶段：解析参数

- `clip_limit`
- `tile_grid_size`

第二阶段：读取图像

- 遍历 `VOC_Dark/images/trainval`
- 遍历 `VOC_Dark/images/test`

第三阶段：执行 CLAHE

- 将 BGR 图像转为 LAB 颜色空间
- 只对亮度通道 L 做增强
- 与原色度通道合并后转回 BGR

第四阶段：保存结果与清单

- 保存增强图到 `VOC_CLAHE`
- 复制标签、标注、split 和元数据
- 写 `clahe_manifest.json`

### A.5.1.3 关键函数

`parse_tile_grid_size()`

- 功能：将字符串形式的 `8x8` 转成 `(8, 8)`
- 关键点：做格式与正整数检查

`apply_clahe_bgr()`

- 功能：执行 CLAHE 增强
- 关键点：
  - 输入必须是 `uint8`
  - 只处理三通道彩色图
  - 只增强亮度通道，不直接破坏色彩结构

### A.5.1.4 实现中的关键点

CLAHE 是经典方法，看起来简单，但真正的关键在于“只增强亮度通道”。如果直接对 RGB 三通道做直方图均衡化，颜色往往会明显失真。

### A.5.1.5 输入输出

输入：

- `datasets/VOC_Dark`

输出：

- `datasets/VOC_CLAHE`
- `clahe_manifest.json`

---

## A.6 前沿增强实现

### A.6.1 文件：`04_zerodce_process.py`

文件位置：

- [04_zerodce_process.py](<C:/CNN/cv_project/scripts/04_zerodce_process.py:1>)

### A.6.1.1 作用

这个脚本负责使用 Zero-DCE++ 官方预训练快照对暗光图像进行推理增强，生成 `VOC_ZDCEPP`，是前沿方案的增强模块。

### A.6.1.2 分阶段说明

第一阶段：定义推理网络

- 手写 Zero-DCE++ 所需的推理网络结构
- 仅保留卷积层和前向逻辑

第二阶段：加载预训练权重

- 加载 `Epoch99.pth`
- 将官方预训练快照映射到当前推理网络

第三阶段：图像预处理

- 使用 OpenCV 读图时得到的是 BGR
- 先转成 RGB
- 再转成 Tensor，并归一化到 `[0, 1]`

第四阶段：执行推理

- 前向传播得到增强结果
- 输出增强图像

第五阶段：后处理与保存

- 将 Tensor 重新映射到 `[0, 255]`
- 变回 `uint8`
- RGB 转回 BGR
- 保存结果到 `VOC_ZDCEPP`

### A.6.1.3 关键函数/类

`ZeroDCEPPInferenceNet`

- 功能：Zero-DCE++ 的纯净推理网络
- 关键点：仅保留推理必须结构，不引入旧版训练逻辑

`tensor_to_bgr_image()`

- 功能：把网络输出张量恢复成 OpenCV 能保存的图像
- 关键点：
  - 正确做值域映射
  - 保证输出是 `uint8`
  - 通道顺序正确

### A.6.1.4 实现中的关键点

这里的关键难点有两个：

1. `BGR / RGB` 通道一致性
2. 不规则尺寸图像的空间对齐

项目中曾出现过由于下采样与上采样不能整除导致的 shape mismatch，因此后续修复成“直接插值回输入尺寸”的写法，以确保奇数尺寸图片也能正常推理。

### A.6.1.5 输入输出

输入：

- `datasets/VOC_Dark`
- `checkpoints/Zero-DCE++/Epoch99.pth`

输出：

- `datasets/VOC_ZDCEPP`

---

## A.7 统一训练、验证与预测

### A.7.1 文件：`eval_pipeline.py`

文件位置：

- [eval_pipeline.py](<C:/CNN/cv_project/eval_pipeline.py:1>)

### A.7.1.1 作用

这个脚本是整个项目的实验总入口，负责统一管理：

- 验证 `val`
- 单图/多图预测 `predict`
- 微调训练 `train`

### A.7.1.2 分阶段说明

第一阶段：解析数据集与模式

- 支持 `dark / clahe / zdcepp / all`
- 支持 `val / predict / train`

第二阶段：读取数据配置

- 根据选择的数据集，定位相应 YAML 与数据根目录

第三阶段：清理 `.cache`

- 删除旧的 `*.cache`
- 避免不同数据集或不同阶段之间读到错误缓存

第四阶段：处理 YAML 路径

- 动态将 YAML 物化为绝对路径版本
- 避免 Ultralytics 将相对路径错误解析到默认 `datasets` 根目录

第五阶段：加载模型并执行模式

- `val`：执行验证并提取 `mAP50` 与 `mAP50-95`
- `predict`：执行推理并保存可视化结果
- `train`：执行微调训练

第六阶段：写结果摘要

- 将关键信息写入 `last_eval_summary.json`

### A.7.1.3 关键函数

`to_yaml_path()`

- 功能：统一路径分隔符
- 关键点：显式替换反斜杠，兼容 Windows 和 Linux

`materialize_dataset_yaml()`

- 功能：把原始 YAML 改写成绝对路径版本
- 关键点：这是整个项目里为规避 Ultralytics 数据路径误读而做的重要工程修复

`remove_dataset_caches()`

- 功能：删除 `.cache`
- 关键点：避免三套数据集互相串台

`ensure_pretrained_model()`

- 功能：限制默认使用 `.pt` 预训练权重
- 关键点：阻止误传 `.yaml` 导致随机初始化训练

### A.7.1.4 实现中的关键点

这个文件承担的是“实验总调度器”的角色。最关键的不是调用模型本身，而是：

- 保证路径不乱
- 保证缓存不串
- 保证不会误从零训练

这些看起来是工程细节，但实际上正是很多实验“看起来能跑、结果却不对”的根源。

### A.7.1.5 输入输出

输入：

- 数据集 YAML
- 对应数据目录
- 模型权重

输出：

- `runs/` 下的训练结果、验证结果、预测结果
- `last_eval_summary.json`

---

## A.8 云端一键执行脚本

### A.8.1 文件：`run_autodl.sh`

文件位置：

- [run_autodl.sh](<C:/CNN/cv_project/run_autodl.sh:1>)

### A.8.1.1 作用

这个脚本用于在 AutoDL 环境中一键串起完整实验流程，减少手动执行多个脚本和重复输命令的成本。

### A.8.1.2 主要阶段

- 环境检查
- 可选安装依赖
- 可选运行测试
- 跑 01、02、03、04
- 跑 `eval_pipeline.py`
- 按开关决定是否训练、是否验证、是否预测

### A.8.1.3 实现中的关键点

这个脚本最大的作用不是“省几条命令”，而是保证实验流程顺序固定、行为可控，特别适合云端复现实验。

---

## A.9 测试文件说明

### A.9.1 文件：`test_step1_scripts.py`

文件位置：

- [test_step1_scripts.py](<C:/CNN/cv_project/tests/test_step1_scripts.py:1>)

作用：

- 对 01、02、03、04、`eval_pipeline.py` 和 `run_autodl.sh` 做基础存在性与关键逻辑测试

重点覆盖：

- VOC 框坐标转换是否裁剪到 `[0, 1]`
- 类别映射是否固定
- 暗光退化是否保持图像类型与形状
- CLAHE 参数解析是否正常
- Zero-DCE++ 是否能恢复图像和处理奇数尺寸
- YAML 路径处理是否跨平台

---

## A.10 配置文件说明

### A.10.1 文件：`dark.yaml`、`clahe.yaml`、`zdcepp.yaml`

文件位置：

- [dark.yaml](<C:/CNN/cv_project/config/dark.yaml:1>)
- [clahe.yaml](<C:/CNN/cv_project/config/clahe.yaml:1>)
- [zdcepp.yaml](<C:/CNN/cv_project/config/zdcepp.yaml:1>)

作用：

- 给 Ultralytics 提供三套独立的数据集描述
- 保证三条数据流完全隔离

关键点：

- 三套 YAML 必须独立存在
- 路径统一使用正斜杠
- 配合 `eval_pipeline.py` 的绝对路径物化机制一起使用

---

## A.11 附件总结

从代码结构上看，本文并不是“只调用现成框架跑结果”，而是围绕暗光检测任务搭建了一套完整的工程闭环。其关键特点可以概括为：

1. 数据准备部分由作者自行实现；
2. 暗光退化与传统增强流程由作者自行实现；
3. Zero-DCE++ 的纯净推理版由作者自行整理和适配；
4. 训练、验证和预测流程通过统一脚本进行封装；
5. 测试与云端自动化脚本保证了实验的稳定性与可复现性。

因此，从课程作业视角看，本项目既具备完整的实验结果，也具备足够清晰的代码实现层次，能够较好地体现作者对暗光目标检测流水线的理解和实现能力。

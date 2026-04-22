# 基于纯 CNN 级联流水线的暗光目标检测实验报告

## 摘要

暗光环境会显著削弱图像亮度、对比度和纹理细节，并进一步降低目标检测模型的识别与定位能力。围绕“暗光图像增强 + 目标检测”这一典型级联思路，本文基于 PASCAL VOC 2007 数据集构建了一套可复现的暗光目标检测实验流程，并设计了两条纯 CNN 检测流水线：基础方案 `CLAHE + YOLOv5su` 与前沿方案 `Zero-DCE++ + YOLO26n`。在数据层面，本文利用离线退化脚本对 VOC 原始图像进行 Gamma 变暗与高斯噪声扰动，生成暗光数据集 `VOC_Dark`；在增强层面，分别使用传统 CLAHE 方法和深度增强模型 Zero-DCE++ 生成 `VOC_CLAHE` 与 `VOC_ZDCEPP`；在检测层面，则基于官方预训练权重进行迁移学习和验证。

实验结果表明，若直接将预训练检测器用于暗光域推理，性能极低；而在增强后的暗光数据上对检测器进行适度微调后，性能得到显著提升。其中，基础方案 `CLAHE + YOLOv5su` 取得了 `mAP50 = 0.7472`、`mAP50-95 = 0.5235` 的结果，在 `mAP50` 指标上最优；前沿方案 `Zero-DCE++ + YOLO26n` 取得了 `mAP50 = 0.7344`、`mAP50-95 = 0.5324` 的结果，在更严格的 `mAP50-95` 指标上略占优势。结果说明，传统增强与深度增强在暗光检测任务中各具优势，而针对暗光域进行适配微调是性能提升的关键步骤。总体而言，本文在不从零训练检测器的前提下，完成了一套完整、可靠、可分析的暗光目标检测课程实验。

**关键词：** 暗光目标检测；CLAHE；Zero-DCE++；YOLOv5u；YOLO26；PASCAL VOC 2007

## 1 引言

目标检测是计算机视觉中的核心任务之一，广泛应用于自动驾驶、视频监控、智能安防和机器人感知等场景。然而，在夜间、弱照明或复杂低照环境下，图像常常出现亮度不足、局部对比度下降、细节纹理缺失以及噪声增强等问题，导致检测模型难以稳定地识别目标类别并准确回归目标边界框。与正常光照场景相比，暗光场景中的目标检测通常存在明显的域偏移问题，即模型的训练分布与测试分布不再一致，这也是预训练检测器直接应用于低照数据时性能急剧下降的主要原因之一。

围绕这一问题，常见的解决思路是将图像增强与目标检测结合起来，构建“先增强、后检测”的级联流水线。增强模块负责提升暗光图像的可见性和局部对比度，而检测模块负责从增强后的图像中提取语义特征并完成分类与定位。从工程视角来看，这种方案具有两个优点：第一，增强器和检测器可以相对独立设计，便于快速搭建实验；第二，可以优先利用公开的官方预训练权重，在有限课程周期内完成一套可运行、可分析、可复现的实验系统。

基于上述考虑，本文以 PASCAL VOC 2007 为基础数据集，通过自编脚本构造暗光退化数据，并设计了两条暗光目标检测流水线。第一条为基础方案 `CLAHE + YOLOv5su`，强调实现简单、运行稳定、工程代价低；第二条为前沿方案 `Zero-DCE++ + YOLO26n`，强调利用现代低照增强网络与更先进检测器组合，在更严格指标下探索潜在优势。本文的目标并不是从零重写检测器本体，而是围绕暗光检测任务的完整实验链路展开，包括数据准备、退化构造、增强处理、推理封装、训练验证和结果可视化等关键环节。

需要特别说明的是，本文采用了“检测器使用官方预训练权重，任务流水线与工程适配逻辑自行实现”的策略。这一策略既保证了实验在课程作业时间内可落地，又能充分体现作者在数据处理、增强实现、推理封装与实验工程化方面的学习成果和实现能力。

## 2 方法

### 2.1 数据集与暗光数据构建

本文使用的数据集为 PASCAL VOC 2007。该数据集包含 20 个目标类别，标注格式为 VOC XML，在目标检测领域具有较高代表性和广泛可比性。为了适配后续 Ultralytics 检测器训练与验证流程，本文首先通过脚本 [01_voc_download_and_yolo.py](<C:/CNN/cv_project/scripts/01_voc_download_and_yolo.py:1>) 对 VOC 2007 数据进行整理，其核心工作包括：

- 使用 `torchvision.datasets.VOCDetection` 或本地压缩包解压方式获取 VOC2007 数据；
- 固定 VOC20 类别顺序，将 XML 中的类别名转换为 `0-19` 的类别索引；
- 将 VOC 的绝对坐标框 `[xmin, ymin, xmax, ymax]` 转换为 YOLO 所需的归一化中心点格式 `[x_center, y_center, width, height]`；
- 对归一化坐标进行裁剪，避免由于浮点计算或图像尺寸处理失误导致的越界问题。

在完成原始数据整理后，本文并未直接在正常光照图像上实验，而是构造了一套离线暗光数据 `VOC_Dark`。具体过程由 [02_degradation.py](<C:/CNN/cv_project/scripts/02_degradation.py:1>) 完成，其退化协议为：

1. 将输入图像转换为 `float32` 并归一化到 `[0, 1]`；
2. 通过 Gamma 变换模拟亮度衰减；
3. 叠加轻度高斯噪声，模拟低照环境下的成像噪声；
4. 最后再裁剪并映射回 `uint8` 图像。

该设计的好处在于，一方面暗光数据与原始 VOC 图像一一对应，便于后续做增强前后对比；另一方面，离线退化比在线随机增强更容易控制实验变量，也更适合课程报告中的可复现实验设定。

### 2.2 基础方案：CLAHE + YOLOv5su

基础方案的整体流程为：

`VOC_Dark -> CLAHE -> YOLOv5su`

其中，增强模块采用传统方法 CLAHE。其实现位于 [03_clahe_process.py](<C:/CNN/cv_project/scripts/03_clahe_process.py:1>)，主要步骤包括：

- 将暗光图像从 BGR 转换到 LAB 颜色空间；
- 仅对亮度通道 `L` 应用 CLAHE；
- 将增强后的亮度通道与原始色度通道重新合并；
- 输出增强结果到 `VOC_CLAHE`。

这种设计避免了直接对 RGB 三通道做直方图均衡化导致的色彩失真问题，因此是暗光增强中的经典基线方案。作为基础方案，它的优点非常明确：实现简单、计算成本低、对硬件和环境依赖较少，而且在很多场景下能够有效提升局部对比度。

在检测器选择上，本文使用 Ultralytics 官方预训练的 `yolov5su.pt` 权重。基础验证阶段先直接推理，随后在 `VOC_CLAHE` 上继续做少量迁移学习。由于 VOC2007 数据量有限，如果从结构文件随机初始化训练，容易出现“白板启动”导致的难收敛问题，因此本文坚持使用官方预训练权重作为训练起点。

### 2.3 前沿方案：Zero-DCE++ + YOLO26n

前沿方案的整体流程为：

`VOC_Dark -> Zero-DCE++ -> YOLO26n`

增强模块采用 Zero-DCE++。与 CLAHE 不同，Zero-DCE++ 是一种基于深度卷积网络的低照增强方法，能够通过学习像素级增强曲线来提升图像亮度和可见性。本文没有直接使用其官方旧版工程整体运行，而是重构出了一份纯净推理版脚本 [04_zerodce_process.py](<C:/CNN/cv_project/scripts/04_zerodce_process.py:1>)，其特点包括：

- 仅保留前向传播所需的卷积结构和权重加载逻辑；
- 使用官方预训练快照 `Epoch99.pth`；
- 显式处理 OpenCV 的 `BGR` 到 PyTorch 常用 `RGB` 的转换；
- 将网络输出张量正确映射回 `uint8` 图像；
- 对奇数尺寸和不可整除缩放场景做空间对齐，避免 shape mismatch。

这种实现方式有两个重要意义。第一，它规避了官方旧工程对历史环境的强依赖，更适合现代 PyTorch 环境；第二，它本身也体现了对模型推理流程的理解，而不是简单运行原仓库脚本。

检测器方面，本文使用官方预训练的 `yolo26n.pt` 权重，并在 `VOC_ZDCEPP` 上进行迁移学习。相较于基础方案，这条前沿链路更符合“深度增强 + 深度检测”的现代实验组合，因此在更严格定位指标上具有更大潜力。

### 2.4 工程实现说明

从工程角度看，本文并非只是在命令行中调用几条现成命令，而是围绕暗光检测任务手动实现了多项关键模块，主要包括：

- 数据下载与 VOC XML 到 YOLO TXT 的标注转换；
- 暗光退化数据的离线生成；
- CLAHE 批量增强处理；
- Zero-DCE++ 纯净推理版重构；
- 统一的验证、预测与训练脚本 [eval_pipeline.py](<C:/CNN/cv_project/eval_pipeline.py:1>)；
- AutoDL 一键执行脚本 [run_autodl.sh](<C:/CNN/cv_project/run_autodl.sh:1>)。

其中，`eval_pipeline.py` 负责统一封装 `val / predict / train` 三类功能，并解决了实验中几个典型工程问题：

- 自动清除数据目录中的 `.cache`，避免不同数据版本互相污染；
- 将数据 YAML 动态物化为绝对路径版本，避免 Ultralytics 误将相对路径解析到错误根目录；
- 强制优先使用 `.pt` 预训练权重，规避随机初始化训练导致的收敛问题。

因此，尽管检测器主干本身使用的是官方预训练模型，但围绕暗光目标检测这一具体任务的数据准备、增强处理、推理封装、评估可视化和实验自动化流程，均由作者独立实现。

## 3 实验设置

### 3.1 实验平台

本实验的主要训练与验证均在 AutoDL 云平台上完成，核心环境如下：

- 平台：AutoDL
- GPU：RTX 5090 32GB
- Python：3.12
- PyTorch：2.8.0
- CUDA：12.8

该平台配置足以支撑 VOC2007 规模下的离线增强、目标检测微调与验证，同时也兼顾了较高的推理速度和较低的实验迭代成本。

### 3.2 训练与验证配置

两条最终采用的主线配置如下。

| 项目 | 基础方案 | 前沿方案 |
|---|---|---|
| 输入数据 | VOC_CLAHE | VOC_ZDCEPP |
| 检测器 | YOLOv5su | YOLO26n |
| 预训练权重 | 是 | 是 |
| 图像尺寸 | 640 | 640 |
| Epoch | 30 | 30 |
| Batch Size | 48 | 64 |
| Workers | 12 | 12 |

在实验过程中，前沿方案还额外尝试了继续追加训练并增大输入尺寸到 `800` 的设定，但该实验没有进一步提升最终性能，因此没有作为最终结果保留。

### 3.3 评价指标

本文使用目标检测中常见的两个指标：

- `mAP50`：IoU 阈值设为 0.5 时的平均精度
- `mAP50-95`：IoU 从 0.5 到 0.95 多阈值平均下的平均精度

其中，`mAP50` 更强调总体检出能力，而 `mAP50-95` 对定位精度要求更高，更能体现模型边界框质量和鲁棒性。

## 4 实验结果与分析

### 4.1 定量结果

两条最终方案的定量结果如下。

| 方案 | mAP50 | mAP50-95 | 结论 |
|---|---:|---:|---|
| CLAHE + YOLOv5su | 0.7472 | 0.5235 | mAP50 最优 |
| Zero-DCE++ + YOLO26n | 0.7344 | 0.5324 | mAP50-95 最优 |

从结果上看，基础方案在 `mAP50` 上略高，说明其在较宽松 IoU 标准下具有更强的总体检出能力；前沿方案在 `mAP50-95` 上略高，说明其在更严格定位标准下边界框质量更稳。两者差距并不大，这一现象反而很有解释价值：传统增强和深度增强并非简单地“谁完全压倒谁”，而是在不同评价维度下各具优势。

### 4.2 无微调与微调对比

无微调与微调后的结果差异如下。

| 方案 | 无微调 mAP50 | 微调后 mAP50 | 无微调 mAP50-95 | 微调后 mAP50-95 |
|---|---:|---:|---:|---:|
| CLAHE + YOLOv5su | 0.0869 | 0.7472 | 0.0694 | 0.5235 |
| Zero-DCE++ + YOLO26n | 0.0828 | 0.7344 | 0.0661 | 0.5324 |

可以看到，若直接将 COCO 预训练检测器用于暗光 VOC 域推理，两条方案的性能都很低，`mAP50` 均不足 `0.09`。而在增强后的暗光数据上进行迁移学习后，性能显著提升。由此可以得出一个十分明确的结论：**微调不是锦上添花，而是暗光目标检测中真正决定性能的关键步骤。**

### 4.3 两条方案的对比分析

基础方案 `CLAHE + YOLOv5su` 的优势在于：

- 流水线更简单；
- 工程代价更低；
- `mAP50` 最高，说明总体检出能力更强；
- 传统增强方法更稳定，参数更直观。

前沿方案 `Zero-DCE++ + YOLO26n` 的优势在于：

- 增强模块更现代；
- 与检测器组合后在 `mAP50-95` 上略优；
- 在部分复杂暗光场景中表现出更高的定位质量和置信度。

从课程作业角度看，这样的结果非常合理。基础方案更适合作为稳定、低成本、可解释的工程基线，而前沿方案更适合作为强调现代方法潜力的扩展方案。

### 4.4 收敛曲线分析

训练收敛曲线建议使用以下两张图：

- [clahe_results.png](<C:/CNN/cv_project/analysis_results/2026-04-22_autodl_ft30/final_pack/curves/clahe_results.png>)
- [zdcepp_results.png](<C:/CNN/cv_project/analysis_results/2026-04-22_autodl_ft30/final_pack/curves/zdcepp_results.png>)

结合训练日志与结果图可以发现：

1. 两条方案的训练与验证损失整体下降，说明训练过程稳定；
2. `mAP50` 与 `mAP50-95` 曲线在后期趋于平稳，说明模型已基本收敛；
3. 基础方案在 30 epoch 内稳定达到较高水平，而前沿方案前期提升更快，后期进入平台区；
4. 追加训练 `zdcepp_ft30_plus20_img800` 后没有进一步带来有效提升，说明继续提高训练成本并不一定能继续带来收益。

### 4.5 PR 曲线、F1 曲线与混淆矩阵分析

推荐插入以下图表：

- [clahe_BoxPR_curve.png](<C:/CNN/cv_project/analysis_results/2026-04-22_autodl_ft30/final_pack/curves/clahe_BoxPR_curve.png>)
- [zdcepp_BoxPR_curve.png](<C:/CNN/cv_project/analysis_results/2026-04-22_autodl_ft30/final_pack/curves/zdcepp_BoxPR_curve.png>)
- [clahe_BoxF1_curve.png](<C:/CNN/cv_project/analysis_results/2026-04-22_autodl_ft30/final_pack/curves/clahe_BoxF1_curve.png>)
- [zdcepp_BoxF1_curve.png](<C:/CNN/cv_project/analysis_results/2026-04-22_autodl_ft30/final_pack/curves/zdcepp_BoxF1_curve.png>)
- [clahe_confusion_matrix_normalized.png](<C:/CNN/cv_project/analysis_results/2026-04-22_autodl_ft30/final_pack/curves/clahe_confusion_matrix_normalized.png>)
- [zdcepp_confusion_matrix_normalized.png](<C:/CNN/cv_project/analysis_results/2026-04-22_autodl_ft30/final_pack/curves/zdcepp_confusion_matrix_normalized.png>)

这些图可以支撑如下分析：

- PR 曲线体现了精确率与召回率之间的平衡关系，两条方案整体都表现出较稳定的检测能力；
- F1 曲线表明，在合适置信度阈值下，两条方案均能达到较好的综合性能；
- 混淆矩阵显示，`car`、`person`、`train`、`horse` 等类别表现较稳定；
- `chair`、`pottedplant`、`bottle` 等类别仍然较难，这与这些类别的尺寸较小、纹理较弱、边界模糊以及暗光环境下的细节损失有关。

### 4.6 增强效果展示

建议在报告中插入 3 组“四联图”，按如下顺序排版：

`Original -> Dark -> CLAHE -> Zero-DCE++`

可选图像来源如下：

- 原图示例目录：[original_demo](<C:/CNN/cv_project/analysis_results/2026-04-22_autodl_ft30/original_demo>)
- 增强图示例目录：[enhance_demo](<C:/CNN/cv_project/analysis_results/2026-04-22_autodl_ft30/final_pack/enhance_demo>)

从增强效果上看：

- `VOC_Dark` 图像整体亮度明显下降，目标轮廓和背景对比减弱；
- CLAHE 更偏向局部对比度增强，因此会让轮廓更清楚，但也容易把噪声一起抬起来；
- Zero-DCE++ 更偏向整体提亮，视觉上更自然，但在部分图像中会带来一定的颜色偏移或颗粒感。

### 4.7 检测结果图展示

建议在报告中放入 4 到 6 张检测结果图，优先从以下目录选择：

- [predict_demo/clahe](<C:/CNN/cv_project/analysis_results/2026-04-22_autodl_ft30/final_pack/predict_demo/clahe>)
- [predict_demo/zdcepp](<C:/CNN/cv_project/analysis_results/2026-04-22_autodl_ft30/final_pack/predict_demo/zdcepp>)

推荐展示方式为：对同一张图做左右对比，左边放基础方案，右边放前沿方案。

## 5 实验结论

### 5.1 结论一：暗光增强与目标检测级联流水线是有效的

实验结果表明，将暗光图像增强与目标检测串联成级联流水线是可行且有效的。无论是传统增强 CLAHE，还是深度增强 Zero-DCE++，在经过检测器微调后都能显著提高暗光场景下的目标检测性能。

### 5.2 结论二：基础方案与前沿方案各有优势

基础方案 `CLAHE + YOLOv5su` 在 `mAP50` 上取得最优结果，说明其在总体检出能力方面更具优势；前沿方案 `Zero-DCE++ + YOLO26n` 在 `mAP50-95` 上略优，说明其在更严格定位标准下具有更好的边界框质量。二者结果接近，体现出传统增强与深度增强在暗光目标检测中并不是简单替代关系，而是分别对应不同的工程侧重点。

### 5.3 结论三：微调是性能提升的关键

相较于直接使用预训练检测器进行暗光推理，在增强后数据上继续进行迁移学习带来了决定性的性能提升。这一结果说明，预训练权重虽然提供了良好的通用先验，但若不进行目标域适配，模型很难在暗光环境下保持良好的检测能力。

## 6 局限性与改进方向

尽管本文两条方案都取得了较好结果，但仍存在一些局限性：

1. 小目标、弱纹理类别仍然较难检测；
2. CLAHE 虽然增强了局部对比度，但也可能放大噪声；
3. Zero-DCE++ 在部分图像上会出现颜色偏移和颗粒增强；
4. 由于实验周期有限，未对更多增强策略和更大检测器规模进行系统消融。

未来可进一步从以下方向改进：

- 引入更强的暗光增强网络；
- 延长训练周期并进一步优化超参数；
- 设计更丰富的暗光退化协议；
- 增加更系统的消融实验与更多指标分析。

## 7 运行平台与关键指令说明

### 7.1 平台说明

正式训练与验证主要在 AutoDL 云服务器完成，平台配置为：

- GPU：RTX 5090 32GB
- Python：3.12
- PyTorch：2.8.0
- CUDA：12.8

### 7.2 关键运行指令

完整标准流程可由 [run_autodl.sh](<C:/CNN/cv_project/run_autodl.sh:1>) 驱动，也可按步骤手动执行。

典型手动流程如下：

```bash
python cv_project/scripts/01_voc_download_and_yolo.py
python cv_project/scripts/02_degradation.py
python cv_project/scripts/03_clahe_process.py
python cv_project/scripts/04_zerodce_process.py --weights cv_project/checkpoints/Zero-DCE++/Epoch99.pth
python cv_project/eval_pipeline.py --dataset clahe --mode train --model cv_project/checkpoints/Ultralytics/yolov5su.pt
python cv_project/eval_pipeline.py --dataset zdcepp --mode train --model cv_project/checkpoints/Ultralytics/yolo26n.pt
python cv_project/eval_pipeline.py --dataset clahe --mode val --model <clahe_best.pt>
python cv_project/eval_pipeline.py --dataset zdcepp --mode val --model <zdcepp_best.pt>
```

AutoDL 一键脚本入口则为：

```bash
bash cv_project/run_autodl.sh
```

## 8 代码说明

本节给出代码文件的总体说明；如果需要查看按文件、按阶段、按关键点展开的详细实现解释，可进一步参考附件文档 [code_appendix.md](<C:/CNN/cv_project/docs/code_appendix.md:1>)。

### 8.1 数据处理代码

- [01_voc_download_and_yolo.py](<C:/CNN/cv_project/scripts/01_voc_download_and_yolo.py:1>)
  - 数据下载、解压、类别映射、标注转换
- [02_degradation.py](<C:/CNN/cv_project/scripts/02_degradation.py:1>)
  - 暗光退化生成

### 8.2 增强代码

- [03_clahe_process.py](<C:/CNN/cv_project/scripts/03_clahe_process.py:1>)
  - CLAHE 批处理增强
- [04_zerodce_process.py](<C:/CNN/cv_project/scripts/04_zerodce_process.py:1>)
  - Zero-DCE++ 纯净推理实现

### 8.3 检测与评估代码

- [eval_pipeline.py](<C:/CNN/cv_project/eval_pipeline.py:1>)
  - 检测训练、验证、预测统一入口

### 8.4 文档与交接文件

- [report_plan.md](<C:/CNN/cv_project/docs/report_plan.md:1>)
  - 报告结构与图表规划
- [report_research_handoff.md](<C:/CNN/cv_project/docs/report_research_handoff.md:1>)
  - 给外部模型或后续写作者的快速交接文档

---

总体而言，本文完成了一套完整的暗光目标检测课程实验：从数据构建、增强实现、检测微调、结果分析到工程自动化均形成闭环。这一工作既具有实验可复现性，也具备较好的课程展示价值。

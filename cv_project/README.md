# CV Project

这套项目是给暗光目标检测作业准备的，主线分成四步：

1. `01_voc_download_and_yolo.py`
   下载 VOC2007，并把标注转成 YOLO 格式。
2. `02_degradation.py`
   把 `VOC_Original` 离线退化成 `VOC_Dark`。
3. `03_clahe_process.py`
   把 `VOC_Dark` 做 CLAHE 增强，生成 `VOC_CLAHE`。
4. `04_zerodce_process.py`
   用纯净版 Zero-DCE++ 推理生成 `VOC_ZDCEPP`。
5. `eval_pipeline.py`
   用 Ultralytics 统一做 `val / predict / train`。

## 推荐运行顺序

```bash
pip install -r cv_project/requirements-autodl.txt
python cv_project/scripts/01_voc_download_and_yolo.py
python cv_project/scripts/02_degradation.py
python cv_project/scripts/03_clahe_process.py
python cv_project/scripts/04_zerodce_process.py --weights cv_project/checkpoints/Zero-DCE++/Epoch99.pth
python cv_project/eval_pipeline.py --dataset clahe --mode val --model yolov5su.pt
python cv_project/eval_pipeline.py --dataset zdcepp --mode val --model yolo26n.pt
```

`Epoch99.pth` 建议放到 `cv_project/checkpoints/Zero-DCE++/Epoch99.pth`。
如果你提前下好了 `VOCtrainval_06-Nov-2007.zip` 和 `VOCtest_06-Nov-2007.zip`，
直接放到 `cv_project/datasets/VOC_Original/raw/`，`01` 脚本会优先本地解压。
如果你也提前下好了检测器权重，建议放到 `cv_project/checkpoints/Ultralytics/`。

## AutoDL 一键脚本

默认先跑“无训练版”全流程：

```bash
bash cv_project/run_autodl.sh
```

如果你只想先跑到推理和验证，不做训练，其实默认就是这样。

如果你想顺手做少量微调：

```bash
RUN_TRAIN=1 EPOCHS=20 bash cv_project/run_autodl.sh
```

如果你这次只想跑 CLAHE 主线，先把 Zero-DCE++ 关掉：

```bash
RUN_ZDCEPP=0 bash cv_project/run_autodl.sh
```

常用开关：

- `INSTALL_DEPS=1`：先装项目依赖，默认开启
- `RUN_TESTS=1`：先跑最小测试，默认开启
- `RUN_ZDCEPP=1`：是否跑 Zero-DCE++ 主线，默认开启
- `RUN_VAL=1`：是否做验证，默认开启
- `RUN_PREDICT=1`：是否额外导出预测可视化
- `RUN_TRAIN=1`：是否做可选微调，默认关闭
- `DEVICE=0`：GPU 卡号，想用 CPU 可以写成 `DEVICE=cpu`

## 目录说明

- `datasets/VOC_Original`
  原始 VOC2007 图像、XML 和 YOLO 标签。
- `datasets/VOC_Dark`
  暗光退化结果。
- `datasets/VOC_CLAHE`
  CLAHE 增强结果。
- `datasets/VOC_ZDCEPP`
  Zero-DCE++ 增强结果。
- `config/*.yaml`
  三套独立数据配置，避免 `.cache` 串台。

## 走本地权重的例子

```bash
python cv_project/eval_pipeline.py --dataset clahe --mode val --model cv_project/checkpoints/Ultralytics/yolov5su.pt
python cv_project/eval_pipeline.py --dataset zdcepp --mode val --model cv_project/checkpoints/Ultralytics/yolo26n.pt
```

## 手写版基础方案

如果你想更强调“自己理解了基础流程”，可以单独跑这份手写版脚本：

```bash
python cv_project/scripts/05_manual_baseline_clahe_yolov5su.py --max-images 10 --device 0
```

这份脚本里手动实现了这些环节：

- 图片遍历和输入输出目录管理
- CLAHE 增强
- YOLO 结果解析
- 边框和类别文字的手工绘制
- 推理结果汇总 `manual_baseline_summary.json`

它仍然使用官方 `yolov5su.pt` 作为检测器权重，但不再依赖 Ultralytics 默认的可视化输出，比较适合放在作业里展示基础学习过程。

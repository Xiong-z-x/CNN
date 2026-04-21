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

#!/usr/bin/env bash
set -euo pipefail

# 这个脚本是给 AutoDL / Linux 云机准备的。
# 目标不是炫技，而是尽量把“第一次上云就能顺着跑完”的成功率拉高一点。

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PIP_BIN="${PIP_BIN:-pip}"

INSTALL_DEPS="${INSTALL_DEPS:-1}"
RUN_TESTS="${RUN_TESTS:-1}"
RUN_ZDCEPP="${RUN_ZDCEPP:-1}"
RUN_VAL="${RUN_VAL:-1}"
RUN_PREDICT="${RUN_PREDICT:-0}"
RUN_TRAIN="${RUN_TRAIN:-0}"

DEVICE="${DEVICE:-0}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-4}"
EPOCHS="${EPOCHS:-20}"
PREDICT_SOURCE="${PREDICT_SOURCE:-}"

CLAHE_MODEL="${CLAHE_MODEL:-${PROJECT_ROOT}/checkpoints/Ultralytics/yolov5su.pt}"
ZDCEPP_MODEL="${ZDCEPP_MODEL:-${PROJECT_ROOT}/checkpoints/Ultralytics/yolo26n.pt}"
ZDCEPP_WEIGHTS="${ZDCEPP_WEIGHTS:-${PROJECT_ROOT}/checkpoints/Zero-DCE++/Epoch99.pth}"

function log_step() {
  echo
  echo "========== $1 =========="
}

function require_python_package() {
  local package_name="$1"
  "${PYTHON_BIN}" -c "import ${package_name}" >/dev/null 2>&1 || {
    echo "缺少 Python 包: ${package_name}"
    exit 1
  }
}

function require_file() {
  local file_path="$1"
  if [[ ! -f "${file_path}" ]]; then
    echo "缺少文件: ${file_path}"
    exit 1
  fi
}

function run_cmd() {
  echo "+ $*"
  "$@"
}

log_step "环境检查"
run_cmd "${PYTHON_BIN}" --version

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  log_step "安装项目依赖"
  run_cmd "${PIP_BIN}" install -r "${PROJECT_ROOT}/requirements-autodl.txt"
fi

# PyTorch 和 torchvision 我故意不在 requirements 里硬绑死版本，
# 因为 AutoDL 镜像经常已经自带 CUDA 版，强装反而容易把环境装乱。
require_python_package "torch"
require_python_package "torchvision"
require_python_package "cv2"
require_python_package "ultralytics"

if [[ "${RUN_TESTS}" == "1" ]]; then
  log_step "运行最小测试"
  run_cmd "${PYTHON_BIN}" -m unittest discover -s "${PROJECT_ROOT}/tests" -p "test_*.py"
fi

log_step "准备 VOC 原始数据和 YOLO 标签"
run_cmd "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/01_voc_download_and_yolo.py"

log_step "生成 VOC_Dark"
run_cmd "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/02_degradation.py"

log_step "生成 VOC_CLAHE"
run_cmd "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/03_clahe_process.py"

if [[ "${RUN_ZDCEPP}" == "1" ]]; then
  log_step "生成 VOC_ZDCEPP"
  require_file "${ZDCEPP_WEIGHTS}"
  run_cmd "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/04_zerodce_process.py" \
    --weights "${ZDCEPP_WEIGHTS}"
else
  echo "跳过 Zero-DCE++ 阶段。"
fi

if [[ "${RUN_VAL}" == "1" ]]; then
  log_step "验证 CLAHE + YOLOv5u"
  require_file "${CLAHE_MODEL}"
  run_cmd "${PYTHON_BIN}" "${PROJECT_ROOT}/eval_pipeline.py" \
    --dataset clahe \
    --mode val \
    --model "${CLAHE_MODEL}" \
    --device "${DEVICE}" \
    --imgsz "${IMGSZ}" \
    --batch "${BATCH}" \
    --workers "${WORKERS}"

  if [[ "${RUN_ZDCEPP}" == "1" ]]; then
    log_step "验证 Zero-DCE++ + YOLO26"
    require_file "${ZDCEPP_MODEL}"
    run_cmd "${PYTHON_BIN}" "${PROJECT_ROOT}/eval_pipeline.py" \
      --dataset zdcepp \
      --mode val \
      --model "${ZDCEPP_MODEL}" \
      --device "${DEVICE}" \
      --imgsz "${IMGSZ}" \
      --batch "${BATCH}" \
      --workers "${WORKERS}"
  fi
fi

if [[ "${RUN_PREDICT}" == "1" ]]; then
  log_step "导出可视化预测结果"
  if [[ -n "${PREDICT_SOURCE}" ]]; then
    run_cmd "${PYTHON_BIN}" "${PROJECT_ROOT}/eval_pipeline.py" \
      --dataset clahe \
      --mode predict \
      --model "${CLAHE_MODEL}" \
      --device "${DEVICE}" \
      --imgsz "${IMGSZ}" \
      --source "${PREDICT_SOURCE}"

    if [[ "${RUN_ZDCEPP}" == "1" ]]; then
      run_cmd "${PYTHON_BIN}" "${PROJECT_ROOT}/eval_pipeline.py" \
        --dataset zdcepp \
        --mode predict \
        --model "${ZDCEPP_MODEL}" \
        --device "${DEVICE}" \
        --imgsz "${IMGSZ}" \
        --source "${PREDICT_SOURCE}"
    fi
  else
    echo "RUN_PREDICT=1 但没给 PREDICT_SOURCE，默认跳过可视化预测。"
  fi
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  log_step "可选微调：CLAHE + YOLOv5u"
  run_cmd "${PYTHON_BIN}" "${PROJECT_ROOT}/eval_pipeline.py" \
    --dataset clahe \
    --mode train \
    --model "${CLAHE_MODEL}" \
    --device "${DEVICE}" \
    --imgsz "${IMGSZ}" \
    --batch "${BATCH}" \
    --workers "${WORKERS}" \
    --epochs "${EPOCHS}" \
    --name "clahe_finetune_e${EPOCHS}"

  if [[ "${RUN_ZDCEPP}" == "1" ]]; then
    log_step "可选微调：Zero-DCE++ + YOLO26"
    run_cmd "${PYTHON_BIN}" "${PROJECT_ROOT}/eval_pipeline.py" \
      --dataset zdcepp \
      --mode train \
      --model "${ZDCEPP_MODEL}" \
      --device "${DEVICE}" \
      --imgsz "${IMGSZ}" \
      --batch "${BATCH}" \
      --workers "${WORKERS}" \
      --epochs "${EPOCHS}" \
      --name "zdcepp_finetune_e${EPOCHS}"
  fi
fi

log_step "全部流程结束"
echo "项目根目录: ${PROJECT_ROOT}"
echo "如果你刚刚只跑了无训练版，现在已经拿到一套可直接写报告的结果了。"

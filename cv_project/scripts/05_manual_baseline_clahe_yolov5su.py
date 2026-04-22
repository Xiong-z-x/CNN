from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_tile_grid_size(raw_value: str) -> tuple[int, int]:
    """把命令行里的 8x8 这种写法转成 tuple。"""
    normalized = raw_value.lower().replace(" ", "")
    if "x" not in normalized:
        raise ValueError("tile-grid-size 要写成 8x8 这种格式。")
    width_str, height_str = normalized.split("x", maxsplit=1)
    width = int(width_str)
    height = int(height_str)
    if width <= 0 or height <= 0:
        raise ValueError("tile-grid-size 里的宽高都得是正整数。")
    return width, height


def iter_source_images(source: Path) -> list[Path]:
    """既支持单张图，也支持整个目录，跑实验时会更顺手。"""
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"没找到输入源：{source}")
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"这个文件后缀不在支持列表里：{source.suffix}")
        return [source]
    return sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def read_image(image_path: Path):
    import cv2
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as pil_image:
        rgb_image = np.array(pil_image.convert("RGB"))
    image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    if image.size == 0:
        raise FileNotFoundError(f"OpenCV 没读到图片：{image_path}")
    return image


def save_image(image_path: Path, image) -> None:
    import cv2
    from PIL import Image

    image_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb_image).save(image_path)


def apply_clahe_bgr(image, clip_limit: float, tile_grid_size: tuple[int, int]):
    import cv2
    import numpy as np

    if image.dtype != np.uint8:
        raise TypeError(f"CLAHE 输入图片必须是 uint8，现在拿到的是 {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"这里只处理 BGR 三通道图像，当前 shape={image.shape}")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)
    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return np.ascontiguousarray(enhanced)


def _tensorlike_to_list(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _resolve_class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)):
        if 0 <= class_id < len(names):
            return str(names[class_id])
    return str(class_id)


def ultralytics_result_to_detections(result, image_shape: tuple[int, int, int]) -> list[dict]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xyxy_list = _tensorlike_to_list(getattr(boxes, "xyxy", None))
    conf_list = _tensorlike_to_list(getattr(boxes, "conf", None))
    cls_list = _tensorlike_to_list(getattr(boxes, "cls", None))
    names = getattr(result, "names", {})

    image_height, image_width = image_shape[:2]
    detections = []
    for xyxy, confidence, class_id_value in zip(xyxy_list, conf_list, cls_list):
        x1, y1, x2, y2 = clip_xyxy_to_image(xyxy, image_width=image_width, image_height=image_height)
        class_id = int(class_id_value)
        detections.append(
            {
                "class_id": class_id,
                "class_name": _resolve_class_name(names, class_id),
                "confidence": float(confidence),
                "xyxy": [x1, y1, x2, y2],
            }
        )
    return detections


def clip_xyxy_to_image(xyxy, image_width: int, image_height: int) -> list[int]:
    x1, y1, x2, y2 = [float(value) for value in xyxy]

    x1 = max(0, min(int(round(x1)), image_width - 1))
    y1 = max(0, min(int(round(y1)), image_height - 1))
    x2 = max(0, min(int(round(x2)), image_width - 1))
    y2 = max(0, min(int(round(y2)), image_height - 1))

    if x2 <= x1:
        x2 = min(image_width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(image_height - 1, y1 + 1)
    return [x1, y1, x2, y2]


def _pick_color(class_id: int) -> tuple[int, int, int]:
    palette = [
        (56, 56, 255),
        (151, 157, 255),
        (31, 112, 255),
        (29, 178, 255),
        (49, 210, 207),
        (10, 249, 72),
        (23, 204, 146),
        (134, 219, 61),
        (52, 147, 26),
        (187, 212, 0),
    ]
    return palette[class_id % len(palette)]


def draw_detections(image, detections: list[dict]):
    import cv2

    canvas = image.copy()
    image_height, image_width = canvas.shape[:2]
    line_thickness = max(2, round(min(image_height, image_width) / 220))
    font_scale = max(0.55, min(image_height, image_width) / 900)
    font_thickness = max(1, line_thickness - 1)

    for detection in detections:
        x1, y1, x2, y2 = detection["xyxy"]
        color = _pick_color(detection["class_id"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=line_thickness)

        label = f'{detection["class_name"]} {detection["confidence"]:.2f}'
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )
        text_top = max(0, y1 - text_height - baseline - 6)
        text_bottom = text_top + text_height + baseline + 6
        text_right = min(image_width - 1, x1 + text_width + 8)
        cv2.rectangle(canvas, (x1, text_top), (text_right, text_bottom), color, thickness=-1)
        cv2.putText(
            canvas,
            label,
            (x1 + 4, text_bottom - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness=font_thickness,
            lineType=cv2.LINE_AA,
        )

    return canvas


def ensure_output_dirs(output_root: Path) -> dict[str, Path]:
    directories = {
        "enhanced": output_root / "enhanced",
        "predictions": output_root / "predictions",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def load_yolo_model(weights: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "当前环境没装 ultralytics，手写版基线没法加载 YOLOv5su。"
            "如果你要真跑推理，先 pip install ultralytics。"
        ) from exc

    return YOLO(weights)


def predict_one_image(model, image, conf: float, iou: float, device: str):
    if not hasattr(model, "predict"):
        raise TypeError("传进来的模型对象缺少 predict()，没法统一走手写推理流程。")
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )
    if isinstance(results, (list, tuple)):
        if not results:
            raise ValueError("模型 predict() 返回了空结果。")
        return results[0]
    return results


def _relative_output_path(image_path: Path, source: Path) -> Path:
    if source.is_file():
        return Path(image_path.name)
    return image_path.resolve().relative_to(source.resolve())


def run_manual_baseline(
    source: Path,
    output_root: Path,
    weights: str,
    device: str,
    conf: float,
    iou: float,
    clip_limit: float,
    tile_grid_size: tuple[int, int],
    model=None,
    max_images: int | None = None,
    save_enhanced: bool = True,
) -> dict:
    source = source.resolve()
    output_root = output_root.resolve()
    image_paths = iter_source_images(source)
    if max_images is not None and max_images > 0:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise ValueError(f"在输入源里一张可处理图片都没找到：{source}")

    model = model or load_yolo_model(weights)
    output_dirs = ensure_output_dirs(output_root)

    image_summaries = []
    total_detections = 0

    for image_path in image_paths:
        original = read_image(image_path)
        enhanced = apply_clahe_bgr(
            image=original,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
        )
        relative_path = _relative_output_path(image_path=image_path, source=source)

        if save_enhanced:
            save_image(output_dirs["enhanced"] / relative_path, enhanced)

        result = predict_one_image(model=model, image=enhanced, conf=conf, iou=iou, device=device)
        detections = ultralytics_result_to_detections(result=result, image_shape=enhanced.shape)
        total_detections += len(detections)

        predicted = draw_detections(enhanced, detections)
        prediction_path = output_dirs["predictions"] / relative_path
        save_image(prediction_path, predicted)

        image_summaries.append(
            {
                "image_name": image_path.name,
                "relative_path": relative_path.as_posix(),
                "enhanced_path": (output_dirs["enhanced"] / relative_path).as_posix() if save_enhanced else None,
                "prediction_path": prediction_path.as_posix(),
                "num_detections": len(detections),
                "detections": detections,
            }
        )

    summary = {
        "source": source.as_posix(),
        "output_root": output_root.as_posix(),
        "weights": str(weights),
        "device": device,
        "conf": conf,
        "iou": iou,
        "clip_limit": clip_limit,
        "tile_grid_size": list(tile_grid_size),
        "num_images": len(image_paths),
        "total_detections": total_detections,
        "images": image_summaries,
    }
    (output_root / "manual_baseline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def build_argparser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="手写版基础方案：先做 CLAHE，再用官方 YOLOv5su 权重跑推理和可视化。",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "datasets" / "VOC_Dark" / "images" / "test",
        help="输入图片源。既支持单张图，也支持整个目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "runs" / "manual_clahe_yolov5su",
        help="手写版基线的输出目录。",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(project_root / "checkpoints" / "Ultralytics" / "yolov5su.pt"),
        help="YOLOv5su 权重路径。",
    )
    parser.add_argument("--device", type=str, default="0", help="设备号，或者写 cpu。")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值。")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS 的 IoU 阈值。")
    parser.add_argument("--clip-limit", type=float, default=2.0, help="CLAHE 的 clip limit。")
    parser.add_argument(
        "--tile-grid-size",
        type=str,
        default="8x8",
        help="CLAHE 的网格大小，比如 8x8。",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="只想快速试跑几张图时可以设成正整数，0 表示全部跑。",
    )
    parser.add_argument(
        "--save-enhanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否把 CLAHE 增强后的中间图也存下来。",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.conf < 0 or args.conf > 1:
        raise ValueError("conf 应该在 [0, 1] 之间。")
    if args.iou <= 0 or args.iou > 1:
        raise ValueError("iou 应该在 (0, 1] 之间。")
    if args.clip_limit <= 0:
        raise ValueError("clip-limit 必须大于 0。")

    tile_grid_size = parse_tile_grid_size(args.tile_grid_size)
    max_images = args.max_images if args.max_images > 0 else None
    summary = run_manual_baseline(
        source=args.source,
        output_root=args.output_root,
        weights=args.weights,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        clip_limit=args.clip_limit,
        tile_grid_size=tile_grid_size,
        max_images=max_images,
        save_enhanced=args.save_enhanced,
    )

    print("手写版基础方案跑完了。")
    print(f'输入源：{summary["source"]}')
    print(f'输出目录：{summary["output_root"]}')
    print(f'处理图片数：{summary["num_images"]}')
    print(f'总检测框数：{summary["total_detections"]}')
    print(f'结果清单：{Path(summary["output_root"]) / "manual_baseline_summary.json"}')


if __name__ == "__main__":
    main()

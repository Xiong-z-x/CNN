from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET_CONFIGS = {
    "dark": {
        "yaml": "dark.yaml",
        "dataset_root": "VOC_Dark",
        "recommended_model": "yolov5su.pt",
    },
    "clahe": {
        "yaml": "clahe.yaml",
        "dataset_root": "VOC_CLAHE",
        "recommended_model": "yolov5su.pt",
    },
    "zdcepp": {
        "yaml": "zdcepp.yaml",
        "dataset_root": "VOC_ZDCEPP",
        "recommended_model": "yolo26n.pt",
    },
}


def to_yaml_path(path: Path) -> str:
    # 这里别偷懒只用 as_posix，在 Linux 上它不会替我们处理“字符串里本来就写死的反斜杠”。
    # 直接统一替换成 /，这样 Windows 和 Linux 都更稳。
    return str(path).replace("\\", "/")


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def get_dataset_info(dataset_name: str, project_root: Path) -> dict:
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"不支持的数据集别名: {dataset_name}")
    config = DATASET_CONFIGS[dataset_name]
    return {
        "name": dataset_name,
        "yaml_path": project_root / "config" / config["yaml"],
        "dataset_root": project_root / "datasets" / config["dataset_root"],
        "recommended_model": config["recommended_model"],
    }


def remove_dataset_caches(dataset_root: Path) -> list[str]:
    removed_files = []
    for cache_path in dataset_root.rglob("*.cache"):
        cache_path.unlink()
        removed_files.append(cache_path.as_posix())
    return removed_files


def ensure_pretrained_model(model_name: str, allow_random_init: bool) -> None:
    model_path = Path(model_name)
    if allow_random_init:
        return
    if model_path.suffix == ".yaml":
        raise ValueError(
            "你现在传的是模型结构 yaml，这会导致随机初始化。"
            "这次默认只允许 .pt 预训练权重，除非你显式加 --allow-random-init。"
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统一跑 YOLO 的 val / predict / train，顺手把 .cache 陷阱也处理掉。"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["dark", "clahe", "zdcepp", "all"],
        help="要评估哪套数据。",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="val",
        choices=["val", "predict", "train"],
        help="运行模式。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="模型权重名或路径。留空时会按数据集选推荐默认值。",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="输入分辨率。")
    parser.add_argument("--device", type=str, default="0", help="设备号或 cpu。")
    parser.add_argument("--batch", type=int, default=16, help="batch size。")
    parser.add_argument("--workers", type=int, default=4, help="dataloader workers。")
    parser.add_argument("--conf", type=float, default=0.25, help="predict 模式的 conf 阈值。")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS 或验证的 IoU 阈值。")
    parser.add_argument("--epochs", type=int, default=20, help="train 模式的 epoch 数。")
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="predict 模式可选输入源。留空就默认跑对应数据集的 test 图像目录。",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=get_project_root() / "runs",
        help="Ultralytics 输出目录。",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="当前实验名。留空时自动生成。",
    )
    parser.add_argument(
        "--save-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="val 模式是否保存 COCO 风格 json。",
    )
    parser.add_argument(
        "--clear-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否先删掉数据集目录下的 .cache。",
    )
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="只有你真想从零开始训，才打开这个开关。",
    )
    return parser


def run_one_dataset(args: argparse.Namespace, dataset_name: str) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("当前环境没装 ultralytics，没法跑 eval_pipeline.py。") from exc

    project_root = get_project_root()
    dataset_info = get_dataset_info(dataset_name=dataset_name, project_root=project_root)
    yaml_path = dataset_info["yaml_path"]
    dataset_root = dataset_info["dataset_root"]
    if not yaml_path.exists():
        raise FileNotFoundError(f"没找到数据配置文件: {yaml_path}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"没找到数据目录: {dataset_root}")

    if args.clear_cache:
        removed_caches = remove_dataset_caches(dataset_root)
    else:
        removed_caches = []

    model_name = args.model or dataset_info["recommended_model"]
    ensure_pretrained_model(model_name=model_name, allow_random_init=args.allow_random_init)
    model = YOLO(model_name)

    experiment_name = args.name or f"{dataset_name}_{args.mode}"
    yaml_path_str = to_yaml_path(yaml_path.resolve())

    if args.mode == "val":
        results = model.val(
            data=yaml_path_str,
            imgsz=args.imgsz,
            device=args.device,
            batch=args.batch,
            workers=args.workers,
            iou=args.iou,
            project=to_yaml_path(args.project.resolve()),
            name=experiment_name,
            save_json=args.save_json,
        )
        metrics = {}
        for attr in ("box",):
            metric_obj = getattr(results, attr, None)
            if metric_obj is not None:
                metrics["map50"] = getattr(metric_obj, "map50", None)
                metrics["map50_95"] = getattr(metric_obj, "map", None)
                break
        return {
            "dataset": dataset_name,
            "mode": args.mode,
            "model": model_name,
            "yaml_path": yaml_path_str,
            "removed_caches": removed_caches,
            "metrics": metrics,
        }

    if args.mode == "predict":
        source = args.source or (dataset_root / "images" / "test")
        results = model.predict(
            source=to_yaml_path(source.resolve()),
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            project=to_yaml_path(args.project.resolve()),
            name=experiment_name,
            save=True,
        )
        return {
            "dataset": dataset_name,
            "mode": args.mode,
            "model": model_name,
            "yaml_path": yaml_path_str,
            "removed_caches": removed_caches,
            "num_results": len(results),
            "source": to_yaml_path(source.resolve()),
        }

    if args.mode == "train":
        results = model.train(
            data=yaml_path_str,
            imgsz=args.imgsz,
            device=args.device,
            batch=args.batch,
            workers=args.workers,
            epochs=args.epochs,
            project=to_yaml_path(args.project.resolve()),
            name=experiment_name,
        )
        return {
            "dataset": dataset_name,
            "mode": args.mode,
            "model": model_name,
            "yaml_path": yaml_path_str,
            "removed_caches": removed_caches,
            "train_artifact_dir": to_yaml_path(Path(results.save_dir).resolve()),
        }

    raise ValueError(f"不支持的 mode: {args.mode}")


def main() -> None:
    args = build_argparser().parse_args()
    project_root = get_project_root()
    args.project.mkdir(parents=True, exist_ok=True)

    dataset_names = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]
    summaries = [run_one_dataset(args=args, dataset_name=name) for name in dataset_names]

    summary_path = project_root / "runs" / "last_eval_summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("评估脚本执行完成。")
    print(f"汇总结果已写入: {summary_path}")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

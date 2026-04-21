from __future__ import annotations

import argparse
import json
import shutil
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
        raise ValueError("CLAHE 的 tile size 必须是正整数。")
    return width, height


def apply_clahe_bgr(image, clip_limit: float, tile_grid_size: tuple[int, int]):
    import cv2
    import numpy as np

    if image.dtype != np.uint8:
        raise TypeError(f"CLAHE 输入图像必须是 uint8，当前是 {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"CLAHE 这里只处理 BGR 三通道图像，当前 shape={image.shape}")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)
    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return np.ascontiguousarray(enhanced)


def read_image(image_path: Path):
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"OpenCV 没读到图片: {image_path}")
    return image


def save_image(image_path: Path, image) -> None:
    import cv2

    image_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(image_path), image)
    if not ok:
        raise IOError(f"图片保存失败: {image_path}")


def iter_images(split_dir: Path) -> list[Path]:
    if not split_dir.exists():
        raise FileNotFoundError(f"没找到图像目录: {split_dir}")
    return sorted(
        path for path in split_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def copy_metadata(source_root: Path, target_root: Path) -> None:
    for folder_name in ("labels", "annotations", "splits"):
        src_dir = source_root / folder_name
        if src_dir.exists():
            shutil.copytree(src_dir, target_root / folder_name, dirs_exist_ok=True)

    for file_name in ("classes.txt", "dataset_summary.json", "degradation_manifest.json"):
        src_file = source_root / file_name
        if src_file.exists():
            (target_root / file_name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, target_root / file_name)


def process_split(
    source_root: Path,
    target_root: Path,
    split_name: str,
    clip_limit: float,
    tile_grid_size: tuple[int, int],
) -> dict:
    image_paths = iter_images(source_root / "images" / split_name)
    target_dir = target_root / "images" / split_name

    for source_image_path in image_paths:
        image = read_image(source_image_path)
        enhanced = apply_clahe_bgr(
            image=image,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
        )
        save_image(target_dir / source_image_path.name, enhanced)

    return {
        "split": split_name,
        "num_images": len(image_paths),
        "clip_limit": clip_limit,
        "tile_grid_size": list(tile_grid_size),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把 VOC_Dark 离线做 CLAHE 增强，生成 VOC_CLAHE。"
    )
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--source-root",
        type=Path,
        default=project_root / "datasets" / "VOC_Dark",
        help="暗光输入目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "datasets" / "VOC_CLAHE",
        help="CLAHE 增强结果输出目录。",
    )
    parser.add_argument("--clip-limit", type=float, default=2.0, help="CLAHE 的 clip limit。")
    parser.add_argument(
        "--tile-grid-size",
        type=str,
        default="8x8",
        help="CLAHE 的网格大小，写成 8x8 这种形式。",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    tile_grid_size = parse_tile_grid_size(args.tile_grid_size)
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()

    if args.clip_limit <= 0:
        raise ValueError("clip-limit 必须大于 0。")
    if not source_root.exists():
        raise FileNotFoundError(f"源数据目录不存在，请先跑 02 脚本: {source_root}")

    copy_metadata(source_root=source_root, target_root=output_root)
    split_summaries = []
    for split_name in ("trainval", "test"):
        split_summaries.append(
            process_split(
                source_root=source_root,
                target_root=output_root,
                split_name=split_name,
                clip_limit=args.clip_limit,
                tile_grid_size=tile_grid_size,
            )
        )

    manifest = {
        "source_root": source_root.as_posix(),
        "output_root": output_root.as_posix(),
        "clip_limit": args.clip_limit,
        "tile_grid_size": list(tile_grid_size),
        "splits": split_summaries,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "clahe_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("VOC_CLAHE 生成完成。")
    print(f"输入目录: {source_root}")
    print(f"输出目录: {output_root}")


if __name__ == "__main__":
    main()

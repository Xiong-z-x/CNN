from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def apply_low_light_degradation(
    image: np.ndarray,
    gamma: float,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """把正常图像退化成暗光图。这里故意写得很直白，后面查 bug 会轻松很多。"""
    if image.dtype != np.uint8:
        raise TypeError(f"输入图像必须是 uint8，当前是 {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"输入图像必须是 HWC 的三通道彩色图，当前 shape={image.shape}")
    if gamma <= 0:
        raise ValueError(f"gamma 必须大于 0，当前是 {gamma}")
    if noise_std < 0:
        raise ValueError(f"noise_std 不能是负数，当前是 {noise_std}")

    # 这一步千万别在 uint8 上直接算，不然噪声和 gamma 会把图像搞得一脸问号。
    image_float = image.astype(np.float32) / 255.0
    dark_image = np.power(image_float, np.float32(gamma)).astype(np.float32)

    if noise_std > 0:
        noise = rng.normal(loc=0.0, scale=noise_std, size=dark_image.shape).astype(np.float32)
        dark_image = dark_image + noise

    dark_image = np.clip(dark_image, 0.0, 1.0)
    return np.rint(dark_image * 255.0).astype(np.uint8)


def build_per_image_rng(
    relative_image_path: Path,
    global_seed: int,
    gamma_min: float,
    gamma_max: float,
    noise_std_min: float,
    noise_std_max: float,
) -> tuple[float, float, np.random.Generator]:
    """给每张图一个稳定随机种子，这样重跑不会今天一套、明天一套。"""
    seed_source = f"{global_seed}:{relative_image_path.as_posix()}"
    digest = hashlib.sha256(seed_source.encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    rng = np.random.default_rng(local_seed)
    gamma = float(rng.uniform(gamma_min, gamma_max))
    noise_std = float(rng.uniform(noise_std_min, noise_std_max))
    return gamma, noise_std, rng


def read_image(image_path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"OpenCV 没读到图片，路径可能有问题: {image_path}")
    return image


def save_image(image_path: Path, image: np.ndarray) -> None:
    import cv2

    image_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(image_path), image)
    if not ok:
        raise IOError(f"图片保存失败: {image_path}")


def iter_images(split_image_dir: Path) -> list[Path]:
    if not split_image_dir.exists():
        raise FileNotFoundError(f"没找到图像目录: {split_image_dir}")
    return sorted(
        path for path in split_image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def copy_metadata(source_root: Path, target_root: Path) -> None:
    for folder_name in ("labels", "annotations", "splits"):
        src_dir = source_root / folder_name
        if src_dir.exists():
            shutil.copytree(src_dir, target_root / folder_name, dirs_exist_ok=True)

    for file_name in ("classes.txt", "dataset_summary.json"):
        src_file = source_root / file_name
        if src_file.exists():
            (target_root / file_name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, target_root / file_name)


def process_split(
    source_root: Path,
    target_root: Path,
    split_name: str,
    gamma_min: float,
    gamma_max: float,
    noise_std_min: float,
    noise_std_max: float,
    seed: int,
) -> dict:
    source_image_dir = source_root / "images" / split_name
    target_image_dir = target_root / "images" / split_name
    image_paths = iter_images(source_image_dir)

    gammas: list[float] = []
    noise_stds: list[float] = []

    for source_image_path in image_paths:
        relative_to_dataset = source_image_path.relative_to(source_root)
        gamma, noise_std, rng = build_per_image_rng(
            relative_image_path=relative_to_dataset,
            global_seed=seed,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            noise_std_min=noise_std_min,
            noise_std_max=noise_std_max,
        )
        image = read_image(source_image_path)
        dark_image = apply_low_light_degradation(
            image=image,
            gamma=gamma,
            noise_std=noise_std,
            rng=rng,
        )
        save_image(target_image_dir / source_image_path.name, dark_image)
        gammas.append(gamma)
        noise_stds.append(noise_std)

    return {
        "split": split_name,
        "num_images": len(image_paths),
        "gamma": {
            "min": min(gammas) if gammas else None,
            "max": max(gammas) if gammas else None,
            "mean": float(np.mean(gammas)) if gammas else None,
        },
        "noise_std": {
            "min": min(noise_stds) if noise_stds else None,
            "max": max(noise_stds) if noise_stds else None,
            "mean": float(np.mean(noise_stds)) if noise_stds else None,
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把 VOC_Original 离线退化成 VOC_Dark，给后面的增强和检测做输入。"
    )
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--source-root",
        type=Path,
        default=project_root / "datasets" / "VOC_Original",
        help="原始 VOC 数据目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "datasets" / "VOC_Dark",
        help="暗光退化数据输出目录。",
    )
    parser.add_argument("--gamma-min", type=float, default=2.2, help="Gamma 下界。")
    parser.add_argument("--gamma-max", type=float, default=2.8, help="Gamma 上界。")
    parser.add_argument(
        "--noise-std-min",
        type=float,
        default=0.01,
        help="高斯噪声标准差下界，注意这里是在 [0,1] 浮点域上取值。",
    )
    parser.add_argument(
        "--noise-std-max",
        type=float,
        default=0.03,
        help="高斯噪声标准差上界，注意这里是在 [0,1] 浮点域上取值。",
    )
    parser.add_argument("--seed", type=int, default=3407, help="全局随机种子。")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.gamma_min <= 0 or args.gamma_max <= 0:
        raise ValueError("gamma 的上下界都必须大于 0。")
    if args.gamma_min > args.gamma_max:
        raise ValueError("gamma-min 不能大于 gamma-max。")
    if args.noise_std_min < 0 or args.noise_std_max < 0:
        raise ValueError("噪声标准差不能是负数。")
    if args.noise_std_min > args.noise_std_max:
        raise ValueError("noise-std-min 不能大于 noise-std-max。")


def main() -> None:
    args = build_argparser().parse_args()
    validate_args(args)

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    splits = ("trainval", "test")

    if not source_root.exists():
        raise FileNotFoundError(f"源数据目录不存在，请先跑 01 脚本: {source_root}")

    copy_metadata(source_root=source_root, target_root=output_root)

    split_summaries = []
    for split_name in splits:
        summary = process_split(
            source_root=source_root,
            target_root=output_root,
            split_name=split_name,
            gamma_min=args.gamma_min,
            gamma_max=args.gamma_max,
            noise_std_min=args.noise_std_min,
            noise_std_max=args.noise_std_max,
            seed=args.seed,
        )
        split_summaries.append(summary)

    manifest = {
        "source_root": source_root.as_posix(),
        "output_root": output_root.as_posix(),
        "seed": args.seed,
        "gamma_range": [args.gamma_min, args.gamma_max],
        "noise_std_range": [args.noise_std_min, args.noise_std_max],
        "splits": split_summaries,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "degradation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("VOC_Dark 离线退化完成。")
    print(f"输入目录: {source_root}")
    print(f"输出目录: {output_root}")
    for summary in split_summaries:
        print(
            f"- {summary['split']}: {summary['num_images']} 张图，"
            f"gamma 均值 {summary['gamma']['mean']:.4f}，"
            f"noise_std 均值 {summary['noise_std']['mean']:.4f}"
        )


if __name__ == "__main__":
    main()

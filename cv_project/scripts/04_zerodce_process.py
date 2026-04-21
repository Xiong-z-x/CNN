from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


class DepthwisePointwiseBlock(nn.Module):
    """这是官方 Zero-DCE++ 的轻量卷积块，名字换了，但权重键位保持兼容。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class ZeroDCEPPInferenceNet(nn.Module):
    """只保留推理真正需要的部分，不掺官方仓库里那些旧训练逻辑。"""

    def __init__(self, scale_factor: int = 12) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        channels = 32

        self.e_conv1 = DepthwisePointwiseBlock(3, channels)
        self.e_conv2 = DepthwisePointwiseBlock(channels, channels)
        self.e_conv3 = DepthwisePointwiseBlock(channels, channels)
        self.e_conv4 = DepthwisePointwiseBlock(channels, channels)
        self.e_conv5 = DepthwisePointwiseBlock(channels * 2, channels)
        self.e_conv6 = DepthwisePointwiseBlock(channels * 2, channels)
        self.e_conv7 = DepthwisePointwiseBlock(channels * 2, 3)

    def enhance(self, x: torch.Tensor, x_r: torch.Tensor) -> torch.Tensor:
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image_1 = x + x_r * (torch.pow(x, 2) - x)
        x = enhance_image_1 + x_r * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image = x + x_r * (torch.pow(x, 2) - x)
        return enhance_image

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")

        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))
        if self.scale_factor != 1:
            x_r = self.upsample(x_r)
        enhanced = self.enhance(x, x_r)
        return enhanced, x_r


def normalize_state_dict(checkpoint) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("权重文件内容不是字典，没法当 state_dict 用。")

    normalized = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        normalized[key.removeprefix("module.")] = value
    return normalized


def load_model(weights_path: Path, device: torch.device, scale_factor: int) -> ZeroDCEPPInferenceNet:
    model = ZeroDCEPPInferenceNet(scale_factor=scale_factor)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = normalize_state_dict(checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Zero-DCE++ 权重加载不完整，"
            f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
        )
    model.to(device)
    model.eval()
    return model


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def prepare_rgb_tensor(bgr_image) -> torch.Tensor:
    import numpy as np

    if bgr_image.dtype != np.uint8:
        raise TypeError(f"输入图像必须是 uint8，当前是 {bgr_image.dtype}")
    try:
        import cv2

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    except ImportError:
        # 本地没装 cv2 时退一步，用通道翻转兜底，正式跑云端还是建议走 cv2。
        rgb_image = np.ascontiguousarray(bgr_image[:, :, ::-1])
    tensor = torch.from_numpy(rgb_image).float() / 255.0
    tensor = tensor.permute(2, 0, 1).contiguous()
    return tensor


def tensor_to_bgr_image(rgb_tensor: torch.Tensor):
    import numpy as np

    if rgb_tensor.ndim != 3 or rgb_tensor.shape[0] != 3:
        raise ValueError(f"输出 tensor 必须是 CHW 且 C=3，当前 shape={tuple(rgb_tensor.shape)}")

    rgb_tensor = rgb_tensor.detach().float().clamp(0.0, 1.0)
    rgb_image = rgb_tensor.mul(255.0).round().byte().cpu().permute(1, 2, 0).numpy()
    rgb_image = np.ascontiguousarray(rgb_image)
    try:
        import cv2

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    except ImportError:
        bgr_image = np.ascontiguousarray(rgb_image[:, :, ::-1])
    return bgr_image


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

    for file_name in (
        "classes.txt",
        "dataset_summary.json",
        "degradation_manifest.json",
        "clahe_manifest.json",
    ):
        src_file = source_root / file_name
        if src_file.exists():
            (target_root / file_name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, target_root / file_name)


def process_split(
    model: ZeroDCEPPInferenceNet,
    device: torch.device,
    source_root: Path,
    target_root: Path,
    split_name: str,
) -> dict:
    image_paths = iter_images(source_root / "images" / split_name)
    target_dir = target_root / "images" / split_name

    with torch.no_grad():
        for source_image_path in image_paths:
            bgr_image = read_image(source_image_path)
            rgb_tensor = prepare_rgb_tensor(bgr_image).unsqueeze(0).to(device)
            enhanced_rgb_tensor, _ = model(rgb_tensor)
            enhanced_bgr = tensor_to_bgr_image(enhanced_rgb_tensor.squeeze(0))
            save_image(target_dir / source_image_path.name, enhanced_bgr)

    return {
        "split": split_name,
        "num_images": len(image_paths),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用纯净版 Zero-DCE++ 推理脚本处理 VOC_Dark，生成 VOC_ZDCEPP。"
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
        default=project_root / "datasets" / "VOC_ZDCEPP",
        help="Zero-DCE++ 增强结果输出目录。",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=project_root / "checkpoints" / "Zero-DCE++" / "Epoch99.pth",
        help="Zero-DCE++ 的预训练权重路径。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="运行设备，默认 auto，会优先上 CUDA。",
    )
    parser.add_argument(
        "--scale-factor",
        type=int,
        default=12,
        help="Zero-DCE++ 默认使用的缩放因子。",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    weights_path = args.weights.resolve()
    device = choose_device(args.device)

    if not source_root.exists():
        raise FileNotFoundError(f"源数据目录不存在，请先跑 02 脚本: {source_root}")
    if not weights_path.exists():
        raise FileNotFoundError(
            "没找到 Zero-DCE++ 权重，请把 Epoch99.pth 放到指定位置后再跑: "
            f"{weights_path}"
        )
    if args.scale_factor <= 0:
        raise ValueError("scale-factor 必须大于 0。")

    model = load_model(weights_path=weights_path, device=device, scale_factor=args.scale_factor)
    copy_metadata(source_root=source_root, target_root=output_root)

    split_summaries = []
    for split_name in ("trainval", "test"):
        split_summaries.append(
            process_split(
                model=model,
                device=device,
                source_root=source_root,
                target_root=output_root,
                split_name=split_name,
            )
        )

    manifest = {
        "source_root": source_root.as_posix(),
        "output_root": output_root.as_posix(),
        "weights": weights_path.as_posix(),
        "device": str(device),
        "scale_factor": args.scale_factor,
        "splits": split_summaries,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "zerodce_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("VOC_ZDCEPP 生成完成。")
    print(f"输入目录: {source_root}")
    print(f"输出目录: {output_root}")
    print(f"使用设备: {device}")


if __name__ == "__main__":
    main()

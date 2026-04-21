from __future__ import annotations

import argparse
import json
import shutil
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterable


# 这里别临时现编类别顺序，VOC 的 id 只要一乱，后面 mAP 基本就直接躺平了。
CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}


def clip_unit(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def class_name_to_id(class_name: str) -> int:
    if class_name not in CLASS_TO_ID:
        raise ValueError(f"发现了不在 VOC20 里的类别: {class_name}")
    class_id = CLASS_TO_ID[class_name]
    if not 0 <= class_id <= 19:
        raise ValueError(f"类别 id 越界了，当前值是 {class_id}")
    return class_id


def voc_box_to_yolo(
    image_width: int,
    image_height: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> tuple[float, float, float, float]:
    """把 VOC 的绝对坐标转成 YOLO 的归一化中心点格式。"""
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"图像宽高不合法: width={image_width}, height={image_height}")
    if xmax < xmin or ymax < ymin:
        raise ValueError(
            f"标注框坐标顺序不对: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}"
        )

    # 这里跟 Ultralytics 官方 VOC 转换脚本保持一致，能少踩一点“看不出来但就是掉点”的坑。
    x_center = ((xmin + xmax) / 2.0 - 1.0) / image_width
    y_center = ((ymin + ymax) / 2.0 - 1.0) / image_height
    box_width = (xmax - xmin) / image_width
    box_height = (ymax - ymin) / image_height

    return (
        clip_unit(x_center),
        clip_unit(y_center),
        clip_unit(box_width),
        clip_unit(box_height),
    )


def read_split_ids(voc_root: Path, split_name: str) -> list[str]:
    split_file = voc_root / "ImageSets" / "Main" / f"{split_name}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"没找到 split 文件: {split_file}")
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_local_voc_archives(raw_root: Path, year: str) -> list[Path]:
    if year != "2007":
        return []
    required_names = [
        "VOCtrainval_06-Nov-2007.zip",
        "VOCtest_06-Nov-2007.zip",
    ]
    archive_paths = [raw_root / name for name in required_names]
    if all(path.exists() for path in archive_paths):
        return archive_paths
    return []


def extract_local_archives(raw_root: Path, archive_paths: list[Path]) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    for archive_path in archive_paths:
        with zipfile.ZipFile(archive_path, mode="r") as zf:
            zf.extractall(raw_root)


def parse_annotation(xml_path: Path, keep_difficult: bool) -> tuple[int, int, list[tuple[int, float, float, float, float]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_node = root.find("size")
    if size_node is None:
        raise ValueError(f"标注文件缺少 size 节点: {xml_path}")

    image_width = int(size_node.findtext("width", default="0"))
    image_height = int(size_node.findtext("height", default="0"))
    labels: list[tuple[int, float, float, float, float]] = []

    for obj in root.findall("object"):
        class_name = obj.findtext("name", default="").strip()
        difficult = int(obj.findtext("difficult", default="0"))
        if difficult == 1 and not keep_difficult:
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            raise ValueError(f"标注文件里有 object 但没 bndbox: {xml_path}")

        xmin = float(bbox.findtext("xmin", default="0"))
        ymin = float(bbox.findtext("ymin", default="0"))
        xmax = float(bbox.findtext("xmax", default="0"))
        ymax = float(bbox.findtext("ymax", default="0"))

        class_id = class_name_to_id(class_name)
        x_center, y_center, box_width, box_height = voc_box_to_yolo(
            image_width=image_width,
            image_height=image_height,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )
        labels.append((class_id, x_center, y_center, box_width, box_height))

    return image_width, image_height, labels


def write_yolo_label(label_path: Path, labels: Iterable[tuple[int, float, float, float, float]]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        for class_id, x_center, y_center, box_width, box_height in labels
    ]
    label_path.write_text("\n".join(lines), encoding="utf-8")


def ensure_voc_downloaded(raw_root: Path, year: str, splits: Iterable[str]) -> None:
    expected_root = raw_root / "VOCdevkit" / f"VOC{year}"
    if expected_root.exists():
        return

    local_archives = find_local_voc_archives(raw_root=raw_root, year=year)
    if local_archives:
        extract_local_archives(raw_root=raw_root, archive_paths=local_archives)
        if expected_root.exists():
            return

    try:
        from torchvision.datasets import VOCDetection
    except ImportError as exc:
        raise ImportError(
            "当前环境没装 torchvision，01 脚本下载 VOC 需要它。"
        ) from exc

    raw_root.mkdir(parents=True, exist_ok=True)
    for split_name in splits:
        VOCDetection(root=raw_root, year=year, image_set=split_name, download=True)


def prepare_structure(output_root: Path) -> None:
    for folder_name in ("images", "labels", "annotations", "splits"):
        (output_root / folder_name).mkdir(parents=True, exist_ok=True)


def copy_and_convert_split(
    voc_root: Path,
    output_root: Path,
    split_name: str,
    keep_difficult: bool,
) -> dict:
    image_ids = read_split_ids(voc_root, split_name)
    image_dir = output_root / "images" / split_name
    label_dir = output_root / "labels" / split_name
    annotation_dir = output_root / "annotations" / split_name

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    class_counter: Counter[str] = Counter()
    empty_label_files = 0

    split_list_path = output_root / "splits" / f"{split_name}.txt"
    relative_image_paths: list[str] = []

    for image_id in image_ids:
        src_image_path = voc_root / "JPEGImages" / f"{image_id}.jpg"
        src_xml_path = voc_root / "Annotations" / f"{image_id}.xml"
        dst_image_path = image_dir / src_image_path.name
        dst_xml_path = annotation_dir / src_xml_path.name
        dst_label_path = label_dir / f"{image_id}.txt"

        if not src_image_path.exists():
            raise FileNotFoundError(f"缺少原始图片: {src_image_path}")
        if not src_xml_path.exists():
            raise FileNotFoundError(f"缺少原始 XML: {src_xml_path}")

        shutil.copy2(src_image_path, dst_image_path)
        shutil.copy2(src_xml_path, dst_xml_path)

        _, _, labels = parse_annotation(src_xml_path, keep_difficult=keep_difficult)
        write_yolo_label(dst_label_path, labels)

        if not labels:
            empty_label_files += 1
        for class_id, *_ in labels:
            class_counter[CLASSES[class_id]] += 1

        relative_image_paths.append(Path("images") / split_name / dst_image_path.name)

    split_list_path.write_text(
        "\n".join(path.as_posix() for path in relative_image_paths),
        encoding="utf-8",
    )

    return {
        "split": split_name,
        "num_images": len(image_ids),
        "num_empty_label_files": empty_label_files,
        "class_histogram": dict(sorted(class_counter.items())),
    }


def write_metadata(output_root: Path, year: str, split_summaries: list[dict], keep_difficult: bool) -> None:
    (output_root / "classes.txt").write_text("\n".join(CLASSES), encoding="utf-8")
    summary = {
        "dataset_name": "PASCAL VOC",
        "year": year,
        "keep_difficult": keep_difficult,
        "classes": CLASSES,
        "splits": split_summaries,
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="下载 VOC2007，并转成后续 YOLO/暗光流水线更顺手的目录结构。"
    )
    default_output_root = Path(__file__).resolve().parents[1] / "datasets" / "VOC_Original"
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root,
        help="VOC 原始数据和 YOLO 标签的输出目录。",
    )
    parser.add_argument(
        "--year",
        type=str,
        default="2007",
        choices=["2007"],
        help="这次作业先只锁定 VOC2007，别把 2012 混进来。",
    )
    parser.add_argument(
        "--keep-difficult",
        action="store_true",
        help="是否保留 VOC 的 difficult 目标，默认不保留，跟 Ultralytics 官方转换保持一致。",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_root = args.output_root.resolve()
    raw_root = output_root / "raw"
    splits = ("trainval", "test")

    prepare_structure(output_root)
    ensure_voc_downloaded(raw_root=raw_root, year=args.year, splits=splits)

    voc_root = raw_root / "VOCdevkit" / f"VOC{args.year}"
    if not voc_root.exists():
        raise FileNotFoundError(f"下载后没找到 VOC 根目录: {voc_root}")

    split_summaries = []
    for split_name in splits:
        summary = copy_and_convert_split(
            voc_root=voc_root,
            output_root=output_root,
            split_name=split_name,
            keep_difficult=args.keep_difficult,
        )
        split_summaries.append(summary)

    write_metadata(
        output_root=output_root,
        year=args.year,
        split_summaries=split_summaries,
        keep_difficult=args.keep_difficult,
    )

    print("VOC2007 下载和 YOLO 标签转换完成。")
    print(f"输出目录: {output_root}")
    for summary in split_summaries:
        print(
            f"- {summary['split']}: {summary['num_images']} 张图，"
            f"空标签 {summary['num_empty_label_files']} 张"
        )


if __name__ == "__main__":
    main()

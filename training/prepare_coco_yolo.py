import argparse
import json
import os
import shutil
from pathlib import Path


def load_coco(json_path: Path) -> dict:
    with json_path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def build_category_map(categories: list[dict]) -> tuple[dict[int, int], list[str]]:
    categories_sorted = sorted(categories, key=lambda item: int(item.get('id', 0)))
    id_to_index: dict[int, int] = {}
    names: list[str] = []
    for idx, category in enumerate(categories_sorted):
        cat_id = int(category.get('id', idx))
        name = str(category.get('name', f'class_{idx}'))
        id_to_index[cat_id] = idx
        names.append(name)
    return id_to_index, names


def normalize_bbox(bbox: list[float], width: float, height: float) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    x_center = (x + w / 2.0) / width
    y_center = (y + h / 2.0) / height
    return x_center, y_center, w / width, h / height


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def write_labels(label_path: Path, lines: list[str]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text('\n'.join(lines), encoding='utf-8')


def convert_split(
    split: str,
    json_path: Path,
    images_dir: Path,
    output_dir: Path,
    id_to_index: dict[int, int],
) -> None:
    coco = load_coco(json_path)
    images = coco.get('images', [])
    annotations = coco.get('annotations', [])

    image_map: dict[int, dict] = {}
    for image in images:
        image_id = int(image.get('id', -1))
        if image_id < 0:
            continue
        image_map[image_id] = image

    labels_by_image: dict[int, list[str]] = {image_id: [] for image_id in image_map.keys()}

    for ann in annotations:
        image_id = int(ann.get('image_id', -1))
        if image_id not in image_map:
            continue
        bbox = ann.get('bbox')
        if not bbox or len(bbox) != 4:
            continue
        category_id = int(ann.get('category_id', -1))
        if category_id not in id_to_index:
            continue

        image_info = image_map[image_id]
        width = float(image_info.get('width', 0))
        height = float(image_info.get('height', 0))
        if width <= 0 or height <= 0:
            continue

        x_center, y_center, w_norm, h_norm = normalize_bbox(bbox, width, height)
        x_center = clamp(x_center, 0.0, 1.0)
        y_center = clamp(y_center, 0.0, 1.0)
        w_norm = clamp(w_norm, 0.0, 1.0)
        h_norm = clamp(h_norm, 0.0, 1.0)

        class_index = id_to_index[category_id]
        labels_by_image[image_id].append(
            f"{class_index} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
        )

    output_images_dir = output_dir / 'images' / split
    output_labels_dir = output_dir / 'labels' / split
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    for image_id, image_info in image_map.items():
        file_name = str(image_info.get('file_name', '')).replace('\\', '/')
        if not file_name:
            continue
        source_path = images_dir / file_name
        target_path = output_images_dir / file_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.exists():
            if not target_path.exists():
                shutil.copy2(source_path, target_path)
        else:
            print(f"Warning: missing image {source_path}")

        label_path = output_labels_dir / Path(file_name).with_suffix('.txt').name
        labels = labels_by_image.get(image_id, [])
        write_labels(label_path, labels)


def write_dataset_yaml(output_dir: Path, names: list[str]) -> Path:
    yaml_path = output_dir / 'dataset.yaml'
    lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for idx, name in enumerate(names):
        lines.append(f"  {idx}: {name}")
    yaml_path.write_text('\n'.join(lines), encoding='utf-8')
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert COCO JSON to YOLO labels.')
    parser.add_argument('--train-json', required=True, help='Path to train COCO JSON')
    parser.add_argument('--val-json', required=True, help='Path to val COCO JSON')
    parser.add_argument('--train-images', required=True, help='Directory with train images')
    parser.add_argument('--val-images', required=True, help='Directory with val images')
    parser.add_argument('--output-dir', required=True, help='Output directory for YOLO dataset')
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_coco = load_coco(Path(args.train_json))
    categories = train_coco.get('categories', [])
    id_to_index, names = build_category_map(categories)

    convert_split('train', Path(args.train_json), Path(args.train_images), output_dir, id_to_index)
    convert_split('val', Path(args.val_json), Path(args.val_images), output_dir, id_to_index)

    yaml_path = write_dataset_yaml(output_dir, names)
    print(f"Dataset ready: {yaml_path}")


if __name__ == '__main__':
    main()

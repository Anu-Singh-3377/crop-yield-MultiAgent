from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "disease"
ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100
USER_AGENT = "crop-yield-MultiAgent-dataset-preparation/1.0"
WHEAT_ARCHIVE_URL = (
    "https://zenodo.org/api/records/24455348/files/train.zip/content"
)


@dataclass(frozen=True)
class DatasetSource:
    crop: str
    dataset: str
    config: str
    excluded_labels: frozenset[str] = frozenset()
    filter_by_label: bool = False


SOURCES = (
    DatasetSource(
        "Rice",
        "Project-AgML/rice_leaf_bacterial_fungal",
        "raw",
    ),
    DatasetSource(
        "Maize",
        "Project-AgML/maize_disease_classification",
        "default",
    ),
    DatasetSource(
        "Soybean",
        "Project-AgML/soybean_leaf_disease_classification",
        "default",
        frozenset({"Root_images"}),
        filter_by_label=True,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download licensed crop images and prepare ImageFolder splits."
    )
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def request_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(5):
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                raise RuntimeError(
                    f"Dataset metadata request returned HTTP {exc.code}: {exc.reason}"
                ) from exc
            if attempt == 4:
                raise RuntimeError(f"Dataset metadata request failed: {exc}") from exc
            retry_after = exc.headers.get("Retry-After")
            delay = 60 if exc.code == 429 else min(10 * (attempt + 1), 60)
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    pass
            print(f"Dataset API returned HTTP {exc.code}; retrying in {delay:g}s")
            time.sleep(delay)
        except (URLError, TimeoutError) as exc:
            if attempt == 4:
                raise RuntimeError(f"Dataset metadata request failed: {exc}") from exc
            time.sleep(2**attempt)
    raise RuntimeError("Dataset metadata request exhausted its retries")


def normalize_label(crop: str, label: str) -> str:
    if "healthy" in label.lower() or label.lower() == "normal":
        disease = "Healthy"
    else:
        disease = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    return f"{crop}___{disease}"


def save_image(source_url: str, destination: Path, image_size: int) -> None:
    request = Request(source_url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urlopen(request, timeout=90) as response:
                image_bytes = response.read()
            with Image.open(BytesIO(image_bytes)) as image:
                image = image.convert("RGB")
                image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
                destination.parent.mkdir(parents=True, exist_ok=True)
                image.save(destination, format="JPEG", quality=90, optimize=True)
            return
        except HTTPError as exc:
            last_error = exc
            if exc.code != 429 and exc.code < 500:
                break
            if attempt < 4:
                retry_after = exc.headers.get("Retry-After")
                delay = 60 if exc.code == 429 else min(10 * (attempt + 1), 60)
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                time.sleep(delay)
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(2**attempt)
    raise RuntimeError(f"Could not download or decode image: {last_error}") from last_error


def get_dataset_rows(
    source: DatasetSource,
    offset: int,
) -> dict[str, Any]:
    query = urlencode({
        "dataset": source.dataset,
        "config": source.config,
        "split": "train",
        "offset": offset,
        "length": PAGE_SIZE,
    })
    return request_json(f"{ROWS_API}?{query}")


def get_first_rows(source: DatasetSource) -> dict[str, Any]:
    query = urlencode({
        "dataset": source.dataset,
        "config": source.config,
        "split": "train",
    })
    return request_json(
        f"https://datasets-server.huggingface.co/first-rows?{query}"
    )


def get_filtered_rows(
    source: DatasetSource,
    label_index: int,
    offset: int,
) -> dict[str, Any]:
    query = urlencode({
        "dataset": source.dataset,
        "config": source.config,
        "split": "train",
        "where": f'"label"={label_index}',
        "offset": offset,
        "length": PAGE_SIZE,
    })
    return request_json(
        f"https://datasets-server.huggingface.co/filter?{query}"
    )


def save_dataset_row_image(
    source: DatasetSource,
    class_name: str,
    row: dict[str, Any],
    image_name: str,
    image_size: int,
) -> Path | None:
    image_data = row.get("image") or {}
    image_url = image_data.get("src")
    if not image_url:
        return None

    destination = (
        DATA_ROOT
        / "source"
        / "hf_images"
        / class_name
        / image_name
    )
    try:
        cache_requires_refresh = not destination.exists()
        if destination.exists():
            with Image.open(destination) as cached_image:
                cached_image.verify()
            with Image.open(destination) as cached_image:
                cache_requires_refresh = max(cached_image.size) < image_size
        if cache_requires_refresh:
            save_image(image_url, destination, image_size)
        return destination
    except (OSError, RuntimeError) as exc:
        if destination.exists():
            destination.unlink()
        print(f"Skipping {source.crop} image {image_name}: {exc}")
        return None


def cached_class_images(class_name: str) -> list[Path]:
    directory = DATA_ROOT / "source" / "hf_images" / class_name
    valid_paths = []
    if not directory.exists():
        return valid_paths
    for path in sorted(directory.glob("*.jpg")):
        try:
            with Image.open(path) as image:
                image.verify()
        except OSError:
            path.unlink()
            continue
        valid_paths.append(path)
    return valid_paths


def download_filtered_source_images(
    source: DatasetSource,
    max_per_class: int,
    image_size: int,
) -> dict[str, list[Path]]:
    first_page = get_first_rows(source)
    label_feature = next(
        feature for feature in first_page["features"] if feature["name"] == "label"
    )
    label_names = label_feature["type"]["names"]
    counts: dict[str, int] = {}
    image_paths: dict[str, list[Path]] = {}

    for label_index, label_name in enumerate(label_names):
        if label_name in source.excluded_labels:
            continue
        class_name = normalize_label(source.crop, label_name)
        cached_paths = cached_class_images(class_name)
        image_paths[class_name] = cached_paths[:max_per_class]
        counts[class_name] = len(image_paths[class_name])
        if counts[class_name] >= max_per_class:
            continue
        offset = 0

        while counts[class_name] < max_per_class:
            page = get_filtered_rows(source, label_index, offset)
            rows = page.get("rows", [])
            if not rows:
                break

            for row_index, indexed_row in enumerate(rows):
                if counts[class_name] >= max_per_class:
                    break
                image_name = (
                    f"label-{label_index}-row-{offset + row_index:06d}.jpg"
                )
                path = save_dataset_row_image(
                    source,
                    class_name,
                    indexed_row["row"],
                    image_name,
                    image_size,
                )
                if path is not None:
                    if path in image_paths[class_name]:
                        continue
                    counts[class_name] += 1
                    image_paths[class_name].append(path)

            offset += len(rows)
            print(
                f"{source.crop} {class_name}: "
                f"{counts[class_name]}/{max_per_class}",
                flush=True,
            )
            total_rows = page.get("num_rows_total")
            if total_rows is not None and offset >= total_rows:
                break

    if all(count >= max_per_class for count in counts.values()):
        print(
            f"{source.crop}: using cached class-balanced images",
            flush=True,
        )
    print(f"{source.crop} samples: {json.dumps(counts, sort_keys=True)}")
    if not counts or any(count < 3 for count in counts.values()):
        raise RuntimeError(
            f"{source.crop} dataset did not provide at least 3 usable images "
            "for every included class."
        )
    return image_paths


def download_source_images(
    source: DatasetSource,
    max_per_class: int,
    image_size: int,
) -> dict[str, list[Path]]:
    if source.filter_by_label:
        return download_filtered_source_images(
            source,
            max_per_class=max_per_class,
            image_size=image_size,
        )

    page = get_first_rows(source)
    label_feature = next(
        feature for feature in page["features"] if feature["name"] == "label"
    )
    label_names = label_feature["type"]["names"]
    counts: dict[str, int] = {}
    image_paths: dict[str, list[Path]] = {}
    for label_name in label_names:
        if label_name in source.excluded_labels:
            continue
        class_name = normalize_label(source.crop, label_name)
        cached_paths = cached_class_images(class_name)
        image_paths[class_name] = cached_paths[:max_per_class]
        counts[class_name] = len(image_paths[class_name])

    if all(count >= max_per_class for count in counts.values()):
        print(
            f"{source.crop}: using cached class-balanced images",
            flush=True,
        )
        return image_paths

    offset = 0
    print(f"{source.crop}: loading {source.dataset}", flush=True)

    while True:
        if offset:
            page = get_dataset_rows(source, offset)

        rows = page.get("rows", [])
        if not rows:
            break

        for row_index, indexed_row in enumerate(rows):
            row = indexed_row["row"]
            label_index = int(row["label"])
            label_name = label_names[label_index]
            if label_name in source.excluded_labels:
                continue
            class_name = normalize_label(source.crop, label_name)
            if counts[class_name] >= max_per_class:
                continue

            image_name = f"row-{offset + row_index:06d}.jpg"
            path = save_dataset_row_image(
                source,
                class_name,
                row,
                image_name,
                image_size,
            )
            if path is not None:
                if path in image_paths[class_name]:
                    continue
                counts[class_name] += 1
                image_paths[class_name].append(path)

        offset += len(rows)
        total_rows = page.get("num_rows_total")
        print(
            f"{source.crop}: scanned {offset}"
            + (f"/{total_rows} rows" if total_rows is not None else " rows"),
            flush=True,
        )
        if all(count >= max_per_class for count in counts.values()):
            break
        if total_rows is not None and offset >= total_rows:
            break

    print(f"{source.crop} samples: {json.dumps(counts, sort_keys=True)}")
    if not counts or any(count < 3 for count in counts.values()):
        raise RuntimeError(
            f"{source.crop} dataset did not provide at least 3 usable images "
            "for every included class."
        )
    return image_paths


def ensure_wheat_archive() -> Path:
    archive_path = DATA_ROOT / "source" / "wheat" / "train.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        try:
            with ZipFile(archive_path) as archive:
                names = set(archive.namelist())
            if any(name.startswith("train/images/") for name in names):
                return archive_path
        except BadZipFile:
            pass

    for attempt in range(5):
        offset = archive_path.stat().st_size if archive_path.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(WHEAT_ARCHIVE_URL, headers=headers)
        try:
            with urlopen(request, timeout=120) as response:
                append = offset > 0 and response.status == 206
                mode = "ab" if append else "wb"
                with archive_path.open(mode) as destination:
                    shutil.copyfileobj(response, destination, length=1024 * 1024)
            with ZipFile(archive_path) as archive:
                names = set(archive.namelist())
            if not any(name.startswith("train/images/") for name in names):
                raise RuntimeError("Downloaded wheat archive has no training images.")
            return archive_path
        except HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                raise RuntimeError(
                    f"Wheat dataset download returned HTTP {exc.code}: {exc.reason}"
                ) from exc
            if attempt == 4:
                raise RuntimeError(f"Wheat dataset download failed: {exc}") from exc
            time.sleep(60 if exc.code == 429 else 2**attempt)
        except (URLError, TimeoutError, BadZipFile) as exc:
            if attempt == 4:
                raise RuntimeError(f"Wheat dataset download failed: {exc}") from exc
            time.sleep(2**attempt)
    raise RuntimeError("Wheat dataset download exhausted its retries")


def prepare_wheat_patches(
    max_per_class: int,
    image_size: int,
) -> dict[str, list[Path]]:
    archive_path = ensure_wheat_archive()
    patch_size = 256
    resized_width = max(image_size * 4, patch_size * 2)
    per_source_limit = 2
    class_names = ("Wheat___Rust", "Wheat___Healthy")
    image_paths = {class_name: [] for class_name in class_names}
    source_dir = DATA_ROOT / "source" / "wheat_patches"

    with ZipFile(archive_path) as archive:
        image_names = sorted(
            name
            for name in archive.namelist()
            if name.startswith("train/images/")
            and name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        masks_by_stem = {
            Path(name).stem: name
            for name in archive.namelist()
            if name.startswith("train/masks/")
            and name.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        if not image_names or not masks_by_stem:
            raise RuntimeError("Wheat archive is missing paired images or masks.")

        for image_name in image_names:
            source_id = Path(image_name).stem
            mask_name = masks_by_stem.get(source_id)
            if not mask_name:
                continue
            with archive.open(image_name) as image_stream:
                with Image.open(image_stream) as source_image:
                    image = source_image.convert("RGB")
            with archive.open(mask_name) as mask_stream:
                with Image.open(mask_stream) as source_mask:
                    mask = source_mask.convert("L")

            resized_height = round(image.height * resized_width / image.width)
            image = image.resize(
                (resized_width, resized_height),
                Image.Resampling.LANCZOS,
            )
            mask = mask.resize(
                (resized_width, resized_height),
                Image.Resampling.NEAREST,
            )
            hsv = np.asarray(image.convert("HSV"))
            per_source_counts = {class_name: 0 for class_name in class_names}
            patch_index = 0

            for top in range(0, resized_height - patch_size + 1, patch_size):
                for left in range(0, resized_width - patch_size + 1, patch_size):
                    image_patch = image.crop((
                        left,
                        top,
                        left + patch_size,
                        top + patch_size,
                    ))
                    mask_patch = np.asarray(mask.crop((
                        left,
                        top,
                        left + patch_size,
                        top + patch_size,
                    )))
                    rust_coverage = float((mask_patch > 127).mean())
                    if rust_coverage >= 0.05:
                        class_name = "Wheat___Rust"
                    elif rust_coverage <= 0.001:
                        hsv_patch = hsv[
                            top:top + patch_size,
                            left:left + patch_size,
                        ]
                        green_coverage = float((
                            (hsv_patch[:, :, 0] >= 30)
                            & (hsv_patch[:, :, 0] <= 115)
                            & (hsv_patch[:, :, 1] >= 30)
                            & (hsv_patch[:, :, 2] >= 30)
                        ).mean())
                        if green_coverage < 0.30:
                            continue
                        class_name = "Wheat___Healthy"
                    else:
                        continue

                    if (
                        len(image_paths[class_name]) >= max_per_class
                        or per_source_counts[class_name] >= per_source_limit
                    ):
                        continue

                    destination = (
                        source_dir
                        / class_name
                        / f"{source_id}-patch-{patch_index:02d}.jpg"
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    image_patch.save(
                        destination,
                        format="JPEG",
                        quality=90,
                        optimize=True,
                    )
                    image_paths[class_name].append(destination)
                    per_source_counts[class_name] += 1
                    patch_index += 1

            if all(
                len(paths) >= max_per_class
                for paths in image_paths.values()
            ):
                break

    print(
        "Wheat experimental patch samples: "
        + json.dumps(
            {name: len(paths) for name, paths in image_paths.items()},
            sort_keys=True,
        )
    )
    if any(len(paths) < 3 for paths in image_paths.values()):
        raise RuntimeError(
            "Wheat masks did not provide at least 3 rust and healthy patches. "
            "Review the source mask format before training."
        )
    return image_paths


def group_id(path: Path) -> str:
    if "-patch-" in path.stem:
        return "patch:" + path.stem.split("-patch-", 1)[0]
    return f"{path.parent.name}:{path.stem}"


def split_dataset(
    images_by_class: dict[str, list[Path]],
    output_dir: Path,
    seed: int,
) -> None:
    split_names = ("train", "val", "test")
    for split_name in split_names:
        (output_dir / split_name).mkdir(parents=True, exist_ok=True)

    source_groups: dict[str, dict[str, list[Path]]] = {}
    for class_name, paths in images_by_class.items():
        for path in paths:
            source_groups.setdefault(group_id(path), {}).setdefault(
                class_name,
                [],
            ).append(path)

    assigned_groups: dict[str, str] = {}
    for class_name in images_by_class:
        source_count = sum(
            class_name in class_paths
            for class_paths in source_groups.values()
        )
        if source_count < 3:
            raise RuntimeError(
                f"{class_name} needs at least 3 distinct source images to split."
            )

    ordered_source_ids = list(source_groups)
    if len(ordered_source_ids) < 3:
        raise RuntimeError("At least 3 distinct source images are needed to split.")

    for attempt in range(100):
        rng = random.Random(seed + attempt)
        shuffled_source_ids = list(ordered_source_ids)
        rng.shuffle(shuffled_source_ids)
        test_count = max(1, round(len(shuffled_source_ids) * 0.15))
        val_count = max(1, round(len(shuffled_source_ids) * 0.15))
        if test_count + val_count >= len(ordered_source_ids):
            test_count = 1
            val_count = 1
        assigned_groups = {
            source_id: "test"
            for source_id in shuffled_source_ids[:test_count]
        }
        assigned_groups.update({
            source_id: "val"
            for source_id in shuffled_source_ids[
                test_count:test_count + val_count
            ]
        })
        assigned_groups.update({
            source_id: "train"
            for source_id in shuffled_source_ids[test_count + val_count:]
        })
        counts = {
            split_name: {class_name: 0 for class_name in images_by_class}
            for split_name in ("train", "val", "test")
        }
        for source_id, class_paths in source_groups.items():
            split_name = assigned_groups[source_id]
            for class_name in class_paths:
                counts[split_name][class_name] += 1
        if all(
            counts[split_name][class_name] > 0
            for split_name in ("train", "val", "test")
            for class_name in images_by_class
        ):
            break
    else:
        raise RuntimeError(
            "Could not create train/validation/test splits containing every class."
        )

    for source_id, class_paths in source_groups.items():
        split_name = assigned_groups[source_id]
        for class_name, paths in class_paths.items():
            target_dir = output_dir / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for path in paths:
                shutil.copy2(path, target_dir / path.name)

    for class_name in images_by_class:
        for split_name in split_names:
            target_dir = output_dir / split_name / class_name
            if not target_dir.exists() or not any(target_dir.iterdir()):
                raise RuntimeError(
                    f"{class_name} has no images in the {split_name} split."
                )


def main() -> None:
    args = parse_args()
    if args.max_per_class < 3:
        raise ValueError("--max-per-class must be at least 3")
    if args.image_size < 64:
        raise ValueError("--image-size must be at least 64")

    output_dir = DATA_ROOT / "processed"
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"{output_dir} already exists. Pass --force to rebuild its splits."
            )
        shutil.rmtree(output_dir)

    images_by_class: dict[str, list[Path]] = {}
    for source in SOURCES:
        images_by_class.update(
            download_source_images(
                source,
                max_per_class=args.max_per_class,
                image_size=args.image_size,
            )
        )
    images_by_class.update(
        prepare_wheat_patches(
            max_per_class=args.max_per_class,
            image_size=args.image_size,
        )
    )

    split_dataset(images_by_class, output_dir, args.seed)
    summary = {
        split_name: {
            class_dir.name: len(list(class_dir.glob("*.jpg")))
            for class_dir in sorted((output_dir / split_name).iterdir())
            if class_dir.is_dir()
        }
        for split_name in ("train", "val", "test")
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from pathlib import Path

from scripts.prepare_disease_dataset import group_id, normalize_label, split_dataset


def test_normalize_label_marks_healthy_classes():
    assert normalize_label("Rice", "Healthy Rice Leaf") == "Rice___Healthy"
    assert normalize_label("Maize", "MLN") == "Maize___MLN"


def test_split_keeps_patches_from_one_source_together(tmp_path: Path):
    images_by_class = {}
    for source_index in range(6):
        for class_name in ("Wheat___Rust", "Wheat___Healthy"):
            source_dir = tmp_path / "source" / class_name
            source_dir.mkdir(parents=True, exist_ok=True)
            image = source_dir / f"field-{source_index}-patch-0.jpg"
            image.write_bytes(b"test")
            images_by_class.setdefault(class_name, []).append(image)

    output_dir = tmp_path / "processed"
    split_dataset(images_by_class, output_dir, seed=7)

    source_splits = {}
    for split_name in ("train", "val", "test"):
        source_splits[split_name] = set()
        for class_name in images_by_class:
            split_paths = list(
                (output_dir / split_name / class_name).glob("*.jpg")
            )
            assert split_paths
            source_splits[split_name].update(group_id(path) for path in split_paths)

    assert not (source_splits["train"] & source_splits["val"])
    assert not (source_splits["train"] & source_splits["test"])
    assert not (source_splits["val"] & source_splits["test"])

    for source_index in range(6):
        source = f"patch:field-{source_index}"
        rust_split = next(
            split_name
            for split_name in source_splits
            if source in {
                group_id(path)
                for path in (output_dir / split_name / "Wheat___Rust").glob("*.jpg")
            }
        )
        healthy_split = next(
            split_name
            for split_name in source_splits
            if source in {
                group_id(path)
                for path in (output_dir / split_name / "Wheat___Healthy").glob("*.jpg")
            }
        )
        assert rust_split == healthy_split

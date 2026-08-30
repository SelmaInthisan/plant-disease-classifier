"""Prepare the official Hugging Face PlantVillage color dataset.

This downloader deliberately does NOT rely on the `datasets` package builder
configuration. Some local `datasets` versions cannot load the repository's
`color` config even though the dataset repository provides it. Instead we use
Hugging Face Hub files directly:
  - data.zip
  - splits/color_train.txt
  - splits/color_test.txt
  - leaf_grouping/leaf-map.json

The official train/test split is preserved. Validation is created only from
the official training split and is grouped by leaf_id to prevent leakage.
"""
import argparse
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

REPO_ID = "mohanty/PlantVillage"
REVISION = "main"
OUT_DIR = Path("data")
SEED = 42
VAL_RATIO = 0.15
CLASS_TO_IDX = {"Diseased": 0, "Healthy": 1}


def binary_label(class_name: str) -> str:
    return "Healthy" if str(class_name).endswith("___healthy") else "Diseased"


def download_repo_file(filename: str) -> Path:
    print(f"Downloading Hugging Face file: {filename}")
    return Path(hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset", revision=REVISION))


def parse_leaf_map(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def image_identifier(filename: str) -> str:
    name = filename.rsplit("/", 1)[-1]
    name = name.replace("_final_masked", "")
    if "___" in name:
        name = name.split("___")[-1]
    name = name.split("copy")[0]
    for ext in (".jpg", ".JPG", ".png", ".PNG"):
        name = name.replace(ext, "")
    return name.strip()


def get_leaf_id(rel_path: str, leaf_map: dict) -> str:
    parts = rel_path.split("/")
    class_name = parts[2] if len(parts) >= 4 else "unknown"
    key = image_identifier(rel_path).lower().strip()
    suggestions = leaf_map.get(key, [])
    if len(suggestions) == 1:
        return suggestions[0]
    for suggestion in suggestions:
        if class_name in suggestion:
            return suggestion
    return f"fallback_{key}"


def read_split_file(path: Path, leaf_map: dict):
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            parts = rel.split("/")
            if len(parts) < 4:
                continue
            class_name = parts[2]
            entries.append({
                "path": rel,
                "class_name": class_name,
                "class": binary_label(class_name),
                "leaf_id": get_leaf_id(rel, leaf_map),
            })
    return entries


def make_leaf_split(train_entries, val_ratio, seed):
    # Only healthy/diseased entries are relevant to this activity.
    entries = [e for e in train_entries if e["class"] in CLASS_TO_IDX]
    leaf_to_class = {}
    for e in entries:
        old = leaf_to_class.get(e["leaf_id"])
        if old is not None and old != e["class"]:
            raise ValueError(f"Leaf {e['leaf_id']} has conflicting binary labels")
        leaf_to_class[e["leaf_id"]] = e["class"]

    leaf_ids = np.array(list(leaf_to_class))
    leaf_labels = np.array([leaf_to_class[x] for x in leaf_ids])
    train_leaf, val_leaf = train_test_split(
        leaf_ids,
        test_size=val_ratio,
        random_state=seed,
        stratify=leaf_labels,
    )
    return set(train_leaf), set(val_leaf)


def extract_entries(zip_path, entries, split_name, max_per_class=None):
    targets = {c: OUT_DIR / split_name / c for c in CLASS_TO_IDX}
    for d in targets.values():
        d.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        for e in entries:
            if max_per_class is not None and counts[e["class"]] >= max_per_class:
                continue
            rel = e["path"]
            # data.zip may have a single top-level directory; resolve robustly.
            candidates = [rel, rel.lstrip("./")]
            member = next((c for c in candidates if c in names), None)
            if member is None:
                suffix = "/" + rel.lstrip("/")
                matches = [n for n in names if n.endswith(suffix)]
                if not matches:
                    raise FileNotFoundError(f"Image not found in data.zip: {rel}")
                member = matches[0]

            counts[e["class"]] += 1
            out = targets[e["class"]] / f"{e['class']}_{counts[e['class']]:06d}.jpg"
            with zf.open(member) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    result = dict(counts)
    print(f"{split_name}: {result}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max_per_class", type=int, default=500,
                        help="Maximum images per class per split. Default 500 gives a compact balanced dataset for CPU training.")
    args = parser.parse_args()

    global OUT_DIR
    OUT_DIR = Path(args.output_dir)
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    print(f"Using official Hugging Face dataset repository: {REPO_ID}")
    print("Using official color/RGB split files directly (no datasets builder config).")

    data_zip = download_repo_file("data.zip")
    train_txt = download_repo_file("splits/color_train.txt")
    test_txt = download_repo_file("splits/color_test.txt")
    leaf_map_file = download_repo_file("leaf_grouping/leaf-map.json")
    leaf_map = parse_leaf_map(leaf_map_file)

    train_all = read_split_file(train_txt, leaf_map)
    test_all = read_split_file(test_txt, leaf_map)
    train_all = [e for e in train_all if e["class"] in CLASS_TO_IDX]
    test_all = [e for e in test_all if e["class"] in CLASS_TO_IDX]

    print(f"Official train entries (Healthy/Diseased): {len(train_all)}")
    print(f"Official test entries  (Healthy/Diseased): {len(test_all)}")

    train_leaf_ids, val_leaf_ids = make_leaf_split(train_all, args.val_ratio, args.seed)
    train_entries = [e for e in train_all if e["leaf_id"] in train_leaf_ids]
    val_entries = [e for e in train_all if e["leaf_id"] in val_leaf_ids]

    train_counts = extract_entries(data_zip, train_entries, "train", args.max_per_class)
    val_counts = extract_entries(data_zip, val_entries, "val", args.max_per_class)
    # Preserve the official test split; do not reshuffle it.
    test_counts = extract_entries(data_zip, test_all, "test", args.max_per_class)

    Path("model").mkdir(exist_ok=True)
    with open("model/class_indices.json", "w", encoding="utf-8") as f:
        json.dump(CLASS_TO_IDX, f, indent=2)
    with open("model/class_names.json", "w", encoding="utf-8") as f:
        json.dump({str(v): k for k, v in CLASS_TO_IDX.items()}, f, indent=2)

    metadata = {
        "dataset": REPO_ID,
        "configuration": "color",
        "official_test_split_used": True,
        "validation_source": "official color_train.txt only",
        "validation_split": args.val_ratio,
        "leaf_id_safe_split": True,
        "seed": args.seed,
        "classes": CLASS_TO_IDX,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "test_counts": test_counts,
    }
    with open("model/dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nDataset preparation complete.")
    print("Official PlantVillage color 80/20 test split retained.")
    print("Validation was created leaf-safely from the official training split.")


if __name__ == "__main__":
    main()

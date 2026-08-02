"""
GeoVision AI
Dataset Exploration Module

This script analyzes the DeepGlobe dataset structure and generates
basic statistics before preprocessing.
"""

from pathlib import Path
import pandas as pd


def count_files(folder: Path):
    """Return number of files inside a folder recursively."""
    if not folder.exists():
        return 0
    return sum(1 for file in folder.rglob("*") if file.is_file())


def print_header(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    dataset_root = Path(input("Enter DeepGlobe dataset path: ").strip())

    if not dataset_root.exists():
        print("\nDataset path not found.")
        return

    print_header("DATASET INFORMATION")

    print(f"Dataset Root : {dataset_root}")

    for folder in ["train", "valid", "test"]:
        folder_path = dataset_root / folder
        print(f"{folder:<10}: {count_files(folder_path)} files")

    print_header("CSV FILES")

    metadata_file = dataset_root / "metadata.csv"
    class_file = dataset_root / "class_dict.csv"

    if metadata_file.exists():
        metadata = pd.read_csv(metadata_file)
        print(f"metadata.csv rows : {len(metadata)}")
        print(metadata.head())

    if class_file.exists():
        classes = pd.read_csv(class_file)
        print("\nClasses:")
        print(classes)

    print_header("Exploration Complete")


if __name__ == "__main__":
    main()
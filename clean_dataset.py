from pathlib import Path
from PIL import Image
import os

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

total_checked = 0
removed_files = 0

for split in ["train", "val", "test"]:
    split_dir = DATASET_DIR / split

    if not split_dir.exists():
        print(f"Missing folder: {split_dir}")
        continue

    for class_folder in split_dir.iterdir():
        if not class_folder.is_dir():
            continue

        print(f"Checking {split}/{class_folder.name}...")

        for file_path in class_folder.iterdir():
            total_checked += 1

            if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                print(f"Removing non-image file: {file_path}")
                file_path.unlink()
                removed_files += 1
                continue

            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception:
                print(f"Removing corrupted image: {file_path}")
                file_path.unlink()
                removed_files += 1

print("Data cleaning completed.")
print(f"Total files checked: {total_checked}")
print(f"Files removed: {removed_files}")
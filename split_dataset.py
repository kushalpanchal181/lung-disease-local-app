import os
import random
import shutil
from pathlib import Path

# Main folders
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_dataset"
OUTPUT_DIR = BASE_DIR / "dataset"

# Split ratio
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Image extensions
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

random.seed(42)

# Check raw dataset exists
if not RAW_DIR.exists():
    raise FileNotFoundError("raw_dataset folder not found. Please create raw_dataset with class folders inside it.")

# Remove old dataset split if it exists
if OUTPUT_DIR.exists():
    print("Removing old dataset folder...")
    shutil.rmtree(OUTPUT_DIR)

# Create train, val, test folders
for split in ["train", "val", "test"]:
    (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

# Get class folders
class_folders = [folder for folder in RAW_DIR.iterdir() if folder.is_dir()]

if not class_folders:
    raise ValueError("No class folders found inside raw_dataset.")

print("Classes found:", [folder.name for folder in class_folders])

for class_folder in class_folders:
    class_name = class_folder.name

    images = [
        file for file in class_folder.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.shuffle(images)

    total_images = len(images)
    train_end = int(total_images * TRAIN_RATIO)
    val_end = train_end + int(total_images * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for split_name, split_images in [
        ("train", train_images),
        ("val", val_images),
        ("test", test_images)
    ]:
        split_class_dir = OUTPUT_DIR / split_name / class_name
        split_class_dir.mkdir(parents=True, exist_ok=True)

        for image in split_images:
            shutil.copy2(image, split_class_dir / image.name)

    print(f"{class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

print("Dataset split completed successfully.")
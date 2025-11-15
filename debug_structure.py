# debug_structure.py - Put this in MLOps-Brain-Tumor folder
from pathlib import Path
import os

print("🔍 DEBUGGING DATA STRUCTURE")
print("=" * 50)

# Use the correct project path
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

print(f"📁 Project root: {PROJECT_ROOT}")
print(f"📁 Data directory: {DATA_DIR}")
print(f"Exists: {DATA_DIR.exists()}")

if DATA_DIR.exists():
    print("\n📂 ALL CONTENTS:")
    for item in DATA_DIR.rglob("*"):  # Recursive search
        if item.is_file():
            # Show relative path from data folder
            rel_path = item.relative_to(DATA_DIR)
            print(f"📄 {rel_path} ({item.stat().st_size} bytes)")
    
    print("\n🔎 FOLDER STRUCTURE:")
    for item in DATA_DIR.rglob(""):
        if item.is_dir():
            rel_path = item.relative_to(DATA_DIR)
            files = list(item.glob("*.*"))
            if files:
                print(f"📁 {rel_path}/ - {len(files)} files")
            else:
                print(f"📁 {rel_path}/ - empty")

print("\n🚨 CHECKING FOR IMAGES:")
image_count = 0
for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
    images = list(DATA_DIR.rglob(ext))
    image_count += len(images)
    if images:
        print(f"Found {len(images)} {ext} files")
        for img in images[:3]:  # Show first 3
            print(f"  → {img.relative_to(DATA_DIR)}")

print(f"\n🎯 TOTAL IMAGES FOUND: {image_count}")